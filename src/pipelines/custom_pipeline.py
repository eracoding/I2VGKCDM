"""
---------------------------------
File: custom_pipeline.py
Author: Ulugbek Shernazarov
Email: u.shernaz4rov@gmail.com
Copyright (c) 2025 Ulugbek Shernazarov. All rights reserved | GitHub: eracoding
Description: A customizable diffusion pipeline for image and video generation that supports various input sources (images, video, audio), interpolation methods, transformation effects, and stabilization techniques. The pipeline enables keyframe-based generation with smooth transitions between prompts, advanced audio-reactive capabilities, and spatial transformations.
---------------------------------
"""
import inspect
import random
from typing import Dict, List, Tuple, Any, Optional, Union
import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from src.preprocess import Preprocessor
from src.helpers import (
    match_color_distribution,
    execute_spatial_transform,
    curve_from_cn_string,
    select_reduction_function,
    read_media_file,
    extract_keyframes,
    spherical_interpolation,
)
from src.pipelines.pipeline_base import PipelineFoundation


class TransformCallback:
    def __init__(self, transform_config, boundary_mode="border"):
        # Initialize animation curves from configuration
        self.scale_factor = transform_config.get("scale", curve_from_cn_string("0:(1.0)"))
        self.x_offset = transform_config.get("x_offset", curve_from_cn_string("0:(0.0)"))
        self.y_offset = transform_config.get("y_offset", curve_from_cn_string("0:(0.0)"))
        self.rotation = transform_config.get("rotation", curve_from_cn_string("0:(0.0)"))
        self.boundary_mode = boundary_mode

    
    def __call__(self, input_image, batch_data):
        frame_num = batch_data["frame_ids"][0]
        image_tensor = ToTensor()(input_image).unsqueeze(0)

        # Collect transform parameters for current frame
        transform_params = {
            "scale": self.scale_factor[frame_num],
            "horizontal_shift": self.x_offset[frame_num],
            "vertical_shift": self.y_offset[frame_num],
            "rotation": self.rotation[frame_num],
        }

        # Apply transformation
        transformed = execute_spatial_transform(
            image_tensor,
            transform_params,
            padding_mode=self.boundary_mode,
        )
        result_image = ToPILImage()(transformed[0])

        return [result_image]
    

class StabilizationCallback:
    def __init__(
        self,
        base_latent=None,
        stability_factor=300,
        memory_decay=0.0,
        iteration_count=1,
        randomization=0.001,
    ):
        self.base_latent = base_latent
        self.stability_factor = stability_factor
        self.memory_decay = memory_decay
        self.randomization = randomization
        self.iteration_count = iteration_count

    def apply(self, iteration, timestep, current_latent):
        # Initialize base latent on first call if needed
        if self.base_latent is None:
            self.base_latent = current_latent

        # Enable gradient calculation for optimization
        current_latent.requires_grad = True

        # Calculate loss as the distance from base latent
        with torch.enable_grad():
            deviation_loss = (current_latent - self.base_latent).pow(2).mean()
            gradient = torch.autograd.grad(deviation_loss, current_latent)[0]

        # Update latent using gradient and noise
        current_latent = current_latent.detach()
        current_latent.add_(-self.stability_factor * gradient)
        current_latent.add_(self.randomization * torch.randn(current_latent.shape).to(current_latent.device))

        # Update base latent with memory decay
        self.base_latent = (self.memory_decay * current_latent) + (
            1.0 - self.memory_decay
        ) * self.base_latent


class CustomPipeline(PipelineFoundation):
    def __init__(
        self,
        diffusion_pipeline,
        text_conditioning,
        compute_device,
        guidance_scale=7.5,
        inference_steps=50,
        strength_curve="0:(0.5)",
        image_height=512,
        image_width=512,
        use_consistent_latent=False,
        use_text_embeddings=True,
        latent_channel_count=4,
        input_image=None,
        audio_source=None,
        audio_type="both",
        audio_reduction_method="max",
        video_source=None,
        use_pil_video_format=False,
        random_seed=42,
        batch_size=1,
        frames_per_second=10,
        negative_conditioning="",
        additional_pipeline_params={},
        interpolation_mode="linear",
        interpolation_settings="",
        transform_settings=None,
        boundary_handling="border",
        stability_strength=350,
        memory_persistence=1.0,
        stability_iterations=1,
        noise_curve="0:(0)",
        enable_color_matching=False,
        image_processors=[],
    ):
        super().__init__(diffusion_pipeline, compute_device, batch_size)
        self.pipeline_parameters = set(inspect.signature(self.diffusion_pipeline).parameters.keys())

            # Store configuration parameters
        self.text_conditioning = text_conditioning
        self.negative_conditioning = negative_conditioning
        self.xl_architecture = hasattr(self.diffusion_pipeline, "text_encoder_2")

        # Generation parameters
        self.use_consistent_latent = use_consistent_latent
        self.use_text_embeddings = use_text_embeddings
        self.latent_channel_count = latent_channel_count
        self.vae_downscale = self.diffusion_pipeline.vae_scale_factor
        self.additional_pipeline_params = additional_pipeline_params

        # Diffusion control parameters
        self.guidance_scale = guidance_scale
        self.inference_steps = inference_steps
        self.strength_curve = curve_from_cn_string(strength_curve)
        self.random_seed = random_seed

        # Environment setup
        self.compute_device = compute_device
        self.random_generator = torch.Generator(self.compute_device).manual_seed(self.random_seed)

        # Video parameters
        self.frames_per_second = frames_per_second

        # Image processing setup
        self.image_processors = image_processors
        if len(self.image_processors) > 1 and self.batch_size > 1:
            raise ValueError(
                f"Multiple image processors can only be used with batch_size=1, not {self.batch_size}"
            )
        self.processor = Preprocessor(self.image_processors)

        # Validate and prepare input sources
        self._validate_inputs(input_image, video_source)

        if input_image is not None:
            resized_image = self._resize_input_image(input_image)
            self.image_width, self.image_height = resized_image.size

            self.reference_image = resized_image.convert("RGB")
            self.source_image = self.processor(resized_image)
        else:
            self.reference_image = self.source_image = None
            self.image_height, self.image_width = image_height, image_width

        # Video setup
        self.video_source = video_source
        self.use_pil_video_format = use_pil_video_format

        self.video_frames = None
        if self.video_source is not None:
            self.video_frames, _, _ = read_media_file(self.video_source)
            _, self.image_height, self.image_width = self.video_frames[0].size()

        # Audio setup
        if audio_source is not None:
            self.audio_data, self.sample_rate = librosa.load(audio_source)
            harmonic, percussive = librosa.effects.hpss(self.audio_data, margin=1.0)

            if audio_type == "percussive":
                self.audio_data = percussive

            if audio_type == "harmonic":
                self.audio_data = harmonic
        else:
            self.audio_data, self.sample_rate = (None, None)

        self.audio_reducer = select_reduction_function(audio_reduction_method)

        # Parse keyframes from text conditioning
        keyframes = extract_keyframes(text_conditioning)
        print("Key Frames: ", keyframes)
        last_frame_idx, _ = max(keyframes, key=lambda x: x[0])
        self.total_frames = last_frame_idx + 1

        # Initialize random seeds for each frame
        random.seed(self.random_seed)
        self.frame_seeds = [
            random.randint(0, 2**64 - 1) for _ in range(self.total_frames)
        ]

        # Prepare interpolation configuration
        interp_config = {
            "mode": interpolation_mode,
            "settings": interpolation_settings,
        }
        
        # Generate latents for each keyframe
        self.frame_latents = self._generate_keyframe_latents(keyframes, interp_config)
        
        # Generate text embeddings if needed
        if self.use_text_embeddings:
            self.embeddings = self._generate_text_embeddings(keyframes, interp_config)
        else:
            self.embeddings = self._extract_text_prompts(keyframes)

        print("Self.embeddings: ", self.embeddings)

        # Set up transform callback if needed
        transform_settings = self._prepare_transform_settings(transform_settings)
        if transform_settings:
            if self.batch_size != 1:
                raise ValueError(
                    f"Transform settings can only be used with batch_size=1, not {self.batch_size}"
                )

            self.transform_callback = TransformCallback(
                transform_settings, boundary_mode=boundary_handling
            )
            self.use_transforms = True
        else:
            self.use_transforms = False

        # Set up stabilization if needed
        self.use_stabilization = stability_strength > 0.0 and self.batch_size == 1
        if self.use_stabilization:
            self.stabilization_callback = StabilizationCallback(
                stability_factor=stability_strength,
                memory_decay=memory_persistence,
                iteration_count=stability_iterations,
            )
        else:
            self.stabilization_callback = None

        # Initialize noise schedule and color matching
        self.noise_schedule = curve_from_cn_string(noise_curve)
        self.enable_color_matching = enable_color_matching
        self.color_match_only = self.enable_color_matching and not self.use_transforms

    def _validate_inputs(self, image_input, video_input):
        """Validate that input sources don't conflict"""
        if image_input is not None and video_input is not None:
            raise ValueError(
                "Cannot use both image_input and video_input simultaneously. Choose one input source."
            )

    def _resize_input_image(self, image_input):
        """Resize input image to be divisible by 8"""
        height, width = image_input.size
        resized_height = height - (height % 8)
        resized_width = width - (width % 8)

        return image_input.resize((resized_height, resized_width), Image.LANCZOS)

    def _prepare_transform_settings(self, transform_settings):
        """Process and validate transform settings"""
        result = {}
        if not transform_settings:
            return result
            
        for key, value in transform_settings.items():
            if len(value) == 0:
                continue
            result[key] = curve_from_cn_string(value)

        return result

    def _generate_audio_interpolation_schedule(
        self, start_frame, end_frame, frames_per_second, audio_data, sample_rate
    ):
        """Generate frame interpolation schedule based on audio energy"""
        frame_count = (end_frame - start_frame) + 1
        frame_duration = sample_rate // frames_per_second

        start_sample = int((start_frame / frames_per_second) * sample_rate)
        end_sample = int((end_frame / frames_per_second) * sample_rate)
        audio_segment = audio_data[start_sample:end_sample]

        # Generate mel spectrogram
        spectrogram = librosa.feature.melspectrogram(
            y=audio_segment, sr=sample_rate, hop_length=frame_duration
        )
        reduced_spec = self.audio_reducer(spectrogram, axis=0)
        normalized_spec = librosa.util.normalize(reduced_spec)

        # Create schedule
        x_coords = np.linspace(0, len(normalized_spec), len(normalized_spec))
        energy_curve = normalized_spec
        cumulative_energy = np.cumsum(normalized_spec)
        cumulative_energy /= cumulative_energy[-1]  # Normalize to 0-1 range

        # Interpolate to match frame count
        target_x = np.linspace(0, len(energy_curve), frame_count)
        interpolated_schedule = np.interp(target_x, x_coords, cumulative_energy)

        return interpolated_schedule

    def _generate_keyframe_latents(self, keyframes, interpolation_config):
        """Generate latent representations for keyframes with interpolation between them"""
        result = {}
        
        # Generate initial latent for first keyframe
        initial_latent = torch.randn(
            (
                1,
                self.latent_channel_count,
                self.image_height // self.vae_downscale,
                self.image_width // self.vae_downscale,
            ),
            dtype=self.diffusion_pipeline.unet.dtype,
            device=self.diffusion_pipeline.device,
            generator=self.random_generator,
        )

        # Process each pair of consecutive keyframes
        for idx, (current_keyframe, next_keyframe) in enumerate(
            zip(keyframes, keyframes[1:])
        ):
            current_frame_idx, _ = current_keyframe
            next_frame_idx, _ = next_keyframe

            # Generate next keyframe latent (or reuse if consistent latent enabled)
            if self.use_consistent_latent:
                next_latent = initial_latent
            else:
                next_latent = torch.randn(
                    (
                        1,
                        self.latent_channel_count,
                        self.image_height // self.vae_downscale,
                        self.image_width // self.vae_downscale,
                    ),
                    dtype=self.diffusion_pipeline.unet.dtype,
                    device=self.diffusion_pipeline.device,
                    generator=self.random_generator.manual_seed(self.frame_seeds[next_frame_idx]),
                )

            # Determine interpolation schedule
            interp_schedule = self._get_interpolation_schedule(
                current_frame_idx,
                next_frame_idx,
                self.frames_per_second,
                interpolation_config,
                self.audio_data,
                self.sample_rate,
            )

            # Generate interpolated latents for each frame between keyframes
            for i, t in enumerate(interp_schedule):
                interpolated_latent = spherical_interpolation(float(t), initial_latent, next_latent)
                result[i + current_frame_idx] = interpolated_latent

            # Update initial latent for next pair
            initial_latent = next_latent

        return result

    def _generate_text_embeddings(self, keyframes, interpolation_config):
        """Generate text embeddings for keyframes with interpolation between them"""
        result = {}

        # Process each pair of consecutive keyframes
        for idx, (current_keyframe, next_keyframe) in enumerate(
            zip(keyframes, keyframes[1:])
        ):
            current_frame_idx, current_prompt = current_keyframe
            next_frame_idx, next_prompt = next_keyframe

            # Generate embeddings for both keyframes
            current_embedding = self.text_to_conditioning(current_prompt)
            next_embedding = self.text_to_conditioning(next_prompt)

            # Get interpolation schedule
            interp_schedule = self._get_interpolation_schedule(
                current_frame_idx,
                next_frame_idx,
                self.frames_per_second,
                interpolation_config,
                self.audio_data,
                self.sample_rate,
            )

            # Generate interpolated embeddings for each frame
            for i, t in enumerate(interp_schedule):
                # Interpolate text embeddings
                interpolated_embed = spherical_interpolation(
                    float(t),
                    current_embedding["text_embeddings"],
                    next_embedding["text_embeddings"],
                )
                frame_result = {"text_embeddings": interpolated_embed}
                
                # Handle pooled embeddings for XL models
                if "pooled_embeddings" in current_embedding:
                    pooled_embed = spherical_interpolation(
                        float(t),
                        current_embedding["pooled_embeddings"],
                        next_embedding["pooled_embeddings"],
                    )
                    frame_result.update({"pooled_embeddings": pooled_embed})
                
                result[i + current_frame_idx] = frame_result

        return result

    def _extract_text_prompts(self, keyframes, use_integers=True, method="linear"):
        """Extract text prompts for each frame by interpolating between keyframes"""
        result = {}
        
        # Create a Series with NaN values for all frames
        frames_series = pd.Series([np.nan for _ in range(self.total_frames)])
        
        # Fill in keyframe values
        for frame_idx, prompt in keyframes:
            frames_series[frame_idx] = prompt

        # Forward-fill to propagate values between keyframes
        frames_series = frames_series.ffill()
        
        # Store result for each frame
        for frame_idx, prompt in enumerate(frames_series):
            result[frame_idx] = prompt

        return result

    def _get_interpolation_schedule(
        self, start_frame, end_frame, fps, interpolation_config, audio_data=None, sample_rate=None
    ):
        """Generate appropriate interpolation schedule based on configuration"""
        # Use audio-based interpolation if audio data is available
        if audio_data is not None:
            return self._generate_audio_interpolation_schedule(
                start_frame, end_frame, fps, audio_data, sample_rate
            )

        # Handle sine-based interpolation
        if interpolation_config["mode"] == "sine":
            return self._generate_sine_schedule(
                start_frame, end_frame, interpolation_config["settings"]
            )

        # Handle custom curve interpolation
        if interpolation_config["mode"] == "curve":
            return self._generate_custom_curve_schedule(
                start_frame, end_frame, interpolation_config["settings"]
            )

        # Default to linear interpolation
        frame_count = (end_frame - start_frame) + 1
        return np.linspace(0, 1, frame_count)

    def _generate_sine_schedule(self, start_frame, end_frame, settings):
        """Generate sine-based interpolation schedule"""
        result = []
        frame_count = (end_frame - start_frame) + 1
        normalized_frames = np.arange(frame_count) / frame_count

        # Parse frequency parameters
        frequencies = settings.split(",")
        if len(frequencies) == 0:
            frequencies = [1.0]
        else:
            frequencies = [float(freq) for freq in frequencies]

        # Generate and sum sine curves with different frequencies
        for frequency in frequencies:
            curve = np.sin(np.pi * normalized_frames * frequency) ** 2
            result.append(curve)

        combined = sum(result)
        normalized = (combined - np.min(combined)) / np.ptp(combined)

        return normalized

    def _generate_custom_curve_schedule(self, start_frame, end_frame, curve_spec):
        """Generate interpolation schedule based on custom curve specification"""
        curve_function = curve_from_cn_string(curve_spec)
        values = []
        
        for frame in range(start_frame, end_frame + 1):
            values.append(curve_function[frame])

        return np.array(values)

    def _frame_batch_generator(self, frames, batch_size):
        """Generate batches of frames for processing"""
        for frame_idx in range(0, len(frames), batch_size):
            start_idx = frame_idx
            end_idx = min(frame_idx + batch_size, len(frames))

            # Get frame indices for this batch
            batch_frames = frames[start_idx:end_idx]
            
            # Get corresponding prompts/embeddings
            if self.use_text_embeddings:
                prompt_batch = [self.embeddings[frame] for frame in batch_frames]
                text_embeddings = [item["text_embeddings"] for item in prompt_batch]
                text_embeddings = torch.cat(text_embeddings, dim=0)

                # Handle pooled embeddings for XL models
                if self.xl_architecture:
                    pooled_embeddings = [item["pooled_embeddings"] for item in prompt_batch]
                    pooled_embeddings = torch.cat(pooled_embeddings, dim=0)
            else:
                # Text prompts mode
                print("Batch Frames: ", batch_frames)
                text_embeddings = [self.embeddings[frame] for frame in batch_frames]

            # Get latents
            latent_batch = [self.frame_latents[frame] for frame in batch_frames]
            latent_batch = torch.cat(latent_batch, dim=0)

            # Get video frames if using video input
            if self.video_frames is not None:
                video_frame_batch = [self.video_frames[frame].unsqueeze(0) for frame in batch_frames]
                if self.use_pil_video_format:
                    video_frame_batch = [ToPILImage()(frame[0]) for frame in video_frame_batch]
                else:
                    video_frame_batch = torch.cat(video_frame_batch, dim=0)
            else:
                video_frame_batch = []

            # Assemble batch data
            batch_data = {
                "prompts": text_embeddings,
                "init_latents": latent_batch,
                "images": video_frame_batch,
                "frame_ids": batch_frames,
            }
            
            print(batch_data)

            # Add pooled embeddings for XL models if needed
            if self.xl_architecture and self.use_text_embeddings:
                batch_data.update({"pooled_prompts": pooled_embeddings})

            yield batch_data

    def _prepare_pipeline_inputs(self, batch_data):
        """Prepare inputs for diffusion pipeline"""
        prompts = batch_data["prompts"]
        latents = batch_data["init_latents"]
        images = batch_data["images"]
        frame_ids = batch_data["frame_ids"]

        # Build pipeline keyword arguments
        pipeline_kwargs = {
            "num_inference_steps": self.inference_steps,
            "guidance_scale": self.guidance_scale,
        }

        # Add height/width if supported
        if "height" in self.pipeline_parameters:
            pipeline_kwargs["height"] = self.image_height

        if "width" in self.pipeline_parameters:
            pipeline_kwargs["width"] = self.image_width

        # Add strength if supported
        if "strength" in self.pipeline_parameters:
            pipeline_kwargs["strength"] = self.strength_curve[frame_ids[0]]

        # Add latents if supported
        if "latents" in self.pipeline_parameters:
            pipeline_kwargs["latents"] = latents

        # Handle prompt embeddings or text prompts
        if "prompt_embeds" in self.pipeline_parameters and self.use_text_embeddings:
            pipeline_kwargs["prompt_embeds"] = prompts
            if self.xl_architecture:
                pipeline_kwargs["pooled_prompt_embeds"] = batch_data["pooled_prompts"]
        elif "prompt" in self.pipeline_parameters and not self.use_text_embeddings:
            pipeline_kwargs["prompt"] = prompts

        # Add negative prompt if supported
        if "negative_prompt" in self.pipeline_parameters:
            pipeline_kwargs["negative_prompt"] = [self.negative_conditioning] * len(prompts)

        # Add image input if supported
        if "image" in self.pipeline_parameters:
            if (self.video_source is not None) and (len(images) != 0):
                # Use current video frame as input
                processed_images = self.processor(images)
                pipeline_kwargs["image"] = processed_images
            elif self.source_image is not None:
                # Use static image as input
                pipeline_kwargs["image"] = self.source_image

        # Add generator if supported
        if "generator" in self.pipeline_parameters:
            pipeline_kwargs["generator"] = self.random_generator

        # Add any additional pipeline arguments
        pipeline_kwargs.update(self.additional_pipeline_params)

        return pipeline_kwargs

    def _get_reference_image(self, current_image):
        """Get reference image for color matching"""
        if self.reference_image is None:
            self.reference_image = current_image

        return self.reference_image

    def _apply_color_matching(self, images):
        """Apply color matching to ensure consistency"""
        # Get reference image
        reference = self._get_reference_image(images[0])
        
        # Match each image to reference
        matched_images = [
            match_color_distribution(image, reference) for image in images
        ]

        return matched_images

    def _apply_transforms(self, images, batch_data):
        """Apply spatial transformations to images"""
        image = images[0]
        image = image.convert("RGB")
        transformed_images = self.transform_callback(image, batch_data)

        return transformed_images

    def _execute_pipeline(self, batch_data):
        """Run the diffusion pipeline with prepared inputs"""
        pipeline_kwargs = self._prepare_pipeline_inputs(batch_data)

        frame_id = batch_data["frame_ids"][0]

        # Set up stabilization if enabled
        if self.use_stabilization:
            noise_level = self.noise_schedule[frame_id]
            self.stabilization_callback.randomization = noise_level
            callback_fn = self.stabilization_callback.apply
            callback_steps = self.stabilization_callback.iteration_count
        else:
            callback_fn = None
            callback_steps = 1

        # Run pipeline
        output = self.diffusion_pipeline(
            **pipeline_kwargs,
            callback=callback_fn,
            callback_steps=callback_steps,
        )

        return output

    def generate_frames(self, frame_indices=None):
        """Generate frames using the configured pipeline"""
        # Use all frames if none specified
        if frame_indices is None:
            frame_indices = list(range(self.total_frames))

        # Create batches of frames
        batch_generator = self._frame_batch_generator(frame_indices, self.batch_size)

        # Process each batch
        for batch_idx, batch in enumerate(batch_generator):
            # Run diffusion pipeline
            output = self._execute_pipeline(batch)
            result_images = output.images

            # Apply color matching if enabled (without transforms)
            if self.color_match_only:
                result_images = self._apply_color_matching(result_images)
                output.images = result_images

            # Return current batch result
            yield output, batch["frame_ids"]

            # Apply transformations if enabled
            if self.use_transforms:
                result_images = self._apply_transforms(result_images, batch)
                if self.enable_color_matching:
                    result_images = self._apply_color_matching(result_images)

                # Update source image for next iteration
                self.source_image = self.processor(result_images)

