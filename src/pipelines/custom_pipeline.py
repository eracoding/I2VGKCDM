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
        pipe,
        text_prompts,
        device,
        guidance_scale=7.5,
        num_inference_steps=50,
        strength="0:(0.5)",
        height=512,
        width=512,
        use_fixed_latent=False,
        use_prompt_embeds=True,
        num_latent_channels=4,
        image_input=None,
        audio_input=None,
        audio_component="both",
        audio_mel_spectogram_reduce="max",
        video_input=None,
        video_use_pil_format=False,
        seed=42,
        batch_size=1,
        fps=10,
        negative_prompts="",
        additional_pipeline_arguments={},
        interpolation_type="linear",
        interpolation_args="",
        motion_args=None,
        padding_mode="border",
        coherence_scale=350,
        coherence_alpha=1.0,
        coherence_steps=1,
        noise_schedule="0:(0)",
        use_color_matching=False,
        preprocess=[],
    ):
        super().__init__(pipe, device, batch_size)
        self.pipe_signature = set(inspect.signature(self.diffusion_pipeline).parameters.keys())

        self.text_prompts = text_prompts
        self.negative_prompts = negative_prompts
        self.is_sdxl = hasattr(self.diffusion_pipeline, "text_encoder_2")

        self.use_fixed_latent = use_fixed_latent
        self.use_prompt_embeds = use_prompt_embeds
        self.num_latent_channels = num_latent_channels
        self.vae_scale_factor = self.diffusion_pipeline.vae_scale_factor
        self.additional_pipeline_arguments = additional_pipeline_arguments

        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.strength = curve_from_cn_string(strength)
        self.seed = seed

        self.device = device
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

        self.fps = fps

        self.preprocess = preprocess
        if len(self.preprocess) > 1 and self.batch_size > 1:
            raise ValueError(
                f"In order to use MultiControlnet",
                f"batch size must be set to 1 but found batch size {self.batch_size}",
            )
        self.preprocessor = Preprocessor(self.preprocess)

        self.check_inputs(image_input, video_input)

        if image_input is not None:
            image_input = self.resize_image_input(image_input)
            self.width, self.height = image_input.size

            self.reference_image = image_input.convert("RGB")
            self.image_input = self.preprocessor(image_input)

        else:
            self.reference_image = self.image_input = None
            self.height, self.width = height, width

        self.video_input = video_input
        self.video_use_pil_format = video_use_pil_format

        self.video_frames = None
        if self.video_input is not None:
            self.video_frames, _, _ = read_media_file(self.video_input)
            _, self.height, self.width = self.video_frames[0].size()

        if audio_input is not None:
            self.audio_array, self.sr = librosa.load(audio_input)
            harmonic, percussive = librosa.effects.hpss(self.audio_array, margin=1.0)

            if audio_component == "percussive":
                self.audio_array = percussive

            if audio_component == "harmonic":
                self.audio_array = harmonic
        else:
            self.audio_array, self.sr = (None, None)

        self.audio_mel_reduce_func = select_reduction_function(audio_mel_spectogram_reduce)

        key_frames = extract_keyframes(text_prompts)
        last_frame, _ = max(key_frames, key=lambda x: x[0])
        self.max_frames = last_frame + 1

        random.seed(self.seed)
        self.seed_schedule = [
            random.randint(0, 18446744073709551615) for i in range(self.max_frames)
        ]

        interpolation_config = {
            "interpolation_type": interpolation_type,
            "interpolation_args": interpolation_args,
        }
        self.init_latents = self.get_init_latents(key_frames, interpolation_config)
        if self.use_prompt_embeds:
            self.prompts = self.get_prompt_embeddings(key_frames, interpolation_config)
        else:
            self.prompts = self.get_prompts(key_frames)

        motion_args = self.prep_animation_args(motion_args)
        if motion_args:
            if self.batch_size != 1:
                raise ValueError(
                    f"In order to use Animation Arguments",
                    f"batch size must be set to 1 but found batch size {self.batch_size}",
                )

            self.motion_callback = MotionCallback(
                motion_args, padding_mode=padding_mode
            )
            self.use_motion = True
        else:
            self.use_motion = False

        self.use_coherence = coherence_scale > 0.0 and self.batch_size == 1
        if self.use_coherence:
            self.coherence_callback = CoherenceCallback(
                coherence_scale=coherence_scale,
                coherence_alpha=coherence_alpha,
                steps=coherence_steps,
            )
        else:
            self.coherence_callback = None

        self.noise_schedule = curve_from_cn_string(noise_schedule)
        self.use_color_matching = use_color_matching
        self.use_color_matching_only = self.use_color_matching and not self.use_motion

    def check_inputs(self, image_input, video_input):
        if image_input is not None and video_input is not None:
            raise ValueError(
                f"Cannot forward both `image_input` and `video_input`. Please make sure to"
                " only forward one of the two."
            )

    def resize_image_input(self, image_input):
        # Resize so image size is divisible by 8
        height, width = image_input.size
        resized_height = height - (height % 8)
        resized_width = width - (width % 8)

        image_input = image_input.resize((resized_height, resized_width), Image.LANCZOS)

        return image_input

    def prep_animation_args(self, animation_args):
        output = {}
        for k, v in animation_args.items():
            if len(v) == 0:
                continue
            output[k] = curve_from_cn_string(v)

        return output

    def get_interpolation_schedule(
        self,
        start_frame,
        end_frame,
        fps,
        interpolation_config,
        audio_array=None,
        sr=None,
    ):
        if audio_array is not None:
            return self.get_interpolation_schedule_from_audio(
                start_frame, end_frame, fps, audio_array, sr
            )

        if interpolation_config["interpolation_type"] == "sine":
            interpolation_args = interpolation_config["interpolation_args"]
            return self.get_sine_interpolation_schedule(
                start_frame, end_frame, interpolation_args
            )

        if interpolation_config["interpolation_type"] == "curve":
            interpolation_args = interpolation_config["interpolation_args"]
            return self.get_curve_interpolation_schedule(
                start_frame, end_frame, interpolation_args
            )

        num_frames = (end_frame - start_frame) + 1

        return np.linspace(0, 1, num_frames)

    def get_sine_interpolation_schedule(
        self, start_frame, end_frame, interpolation_args
    ):
        output = []
        num_frames = (end_frame - start_frame) + 1
        frames = np.arange(num_frames) / num_frames

        interpolation_args = interpolation_args.split(",")
        if len(interpolation_args) == 0:
            interpolation_args = [1.0]
        else:
            interpolation_args = list(map(lambda x: float(x), interpolation_args))

        for frequency in interpolation_args:
            curve = np.sin(np.pi * frames * frequency) ** 2
            output.append(curve)

        schedule = sum(output)
        schedule = (schedule - np.min(schedule)) / np.ptp(schedule)

        return schedule

    def get_interpolation_schedule_from_audio(
        self, start_frame, end_frame, fps, audio_array, sr
    ):
        num_frames = (end_frame - start_frame) + 1
        frame_duration = sr // fps

        start_sample = int((start_frame / fps) * sr)
        end_sample = int((end_frame / fps) * sr)
        audio_slice = audio_array[start_sample:end_sample]

        # from https://aiart.dev/posts/sd-music-videos/sd_music_videos.html
        spec = librosa.feature.melspectrogram(
            y=audio_slice, sr=sr, hop_length=frame_duration
        )
        spec = self.audio_mel_reduce_func(spec, axis=0)
        spec_norm = librosa.util.normalize(spec)

        schedule_x = np.linspace(0, len(spec_norm), len(spec_norm))
        schedule_y = spec_norm
        schedule_y = np.cumsum(spec_norm)
        schedule_y /= schedule_y[-1]

        resized_schedule = np.linspace(0, len(schedule_y), num_frames)
        interp_schedule = np.interp(resized_schedule, schedule_x, schedule_y)

        return interp_schedule

    def get_curve_interpolation_schedule(
        self, start_frame, end_frame, interpolation_args
    ):
        curve = curve_from_cn_string(interpolation_args)
        curve_params = []
        for frame in range(start_frame, end_frame + 1):
            curve_params.append(curve[frame])

        return np.array(curve_params)

    @torch.no_grad()
    def get_prompt_embeddings(self, key_frames, interpolation_config):
        output = {}

        for idx, (start_key_frame, end_key_frame) in enumerate(
            zip(key_frames, key_frames[1:])
        ):
            start_frame, start_prompt = start_key_frame
            end_frame, end_prompt = end_key_frame

            start_prompt_embeds = self.prompt_to_embedding(start_prompt)
            end_prompt_embeds = self.prompt_to_embedding(end_prompt)

            interp_schedule = self.get_interpolation_schedule(
                start_frame,
                end_frame,
                self.fps,
                interpolation_config,
                self.audio_array,
                self.sr,
            )

            for i, t in enumerate(interp_schedule):
                prompt_embed = spherical_interpolation(
                    float(t),
                    start_prompt_embeds["text_embeddings"],
                    end_prompt_embeds["text_embeddings"],
                )
                output[i + start_frame] = {"text_embeddings": prompt_embed}
                if "pooled_embeddings" in start_prompt_embeds:
                    pooled_embed = spherical_interpolation(
                        float(t),
                        start_prompt_embeds["pooled_embeddings"],
                        end_prompt_embeds["pooled_embeddings"],
                    )
                    output[i + start_frame].update({"pooled_embeddings": pooled_embed})

        return output

    def get_prompts(self, key_frames, integer=True, method="linear"):
        output = {}
        key_frame_series = pd.Series([np.nan for a in range(self.max_frames)])
        for frame_idx, prompt in key_frames:
            key_frame_series[frame_idx] = prompt

        key_frame_series = key_frame_series.ffill()
        for frame_idx, prompt in enumerate(key_frame_series):
            output[frame_idx] = prompt

        return output

    @torch.no_grad()
    def get_init_latents(self, key_frames, interpolation_config):
        output = {}
        start_latent = torch.randn(
            (
                1,
                self.num_latent_channels,
                self.height // self.vae_scale_factor,
                self.width // self.vae_scale_factor,
            ),
            dtype=self.diffusion_pipeline.unet.dtype,
            device=self.diffusion_pipeline.device,
            generator=self.generator,
        )

        for idx, (start_key_frame, end_key_frame) in enumerate(
            zip(key_frames, key_frames[1:])
        ):
            start_frame, _ = start_key_frame
            end_frame, _ = end_key_frame

            end_latent = (
                start_latent
                if self.use_fixed_latent
                else torch.randn(
                    (
                        1,
                        self.num_latent_channels,
                        self.height // self.vae_scale_factor,
                        self.width // self.vae_scale_factor,
                    ),
                    dtype=self.diffusion_pipeline.unet.dtype,
                    device=self.diffusion_pipeline.device,
                    generator=self.generator.manual_seed(self.seed_schedule[end_frame]),
                )
            )

            interp_schedule = self.get_interpolation_schedule(
                start_frame,
                end_frame,
                self.fps,
                interpolation_config,
                self.audio_array,
                self.sr,
            )

            for i, t in enumerate(interp_schedule):
                latents = spherical_interpolation(float(t), start_latent, end_latent)
                output[i + start_frame] = latents

            start_latent = end_latent

        return output

    def batch_generator(self, frames, batch_size):
        for frame_idx in range(0, len(frames), batch_size):
            start = frame_idx
            end = frame_idx + batch_size

            frame_batch = frames[start:end]
            prompts_batch = list(map(lambda x: self.prompts[x], frame_batch))
            if self.use_prompt_embeds:
                prompts = list(map(lambda x: x["text_embeddings"], prompts_batch))
                prompts = torch.cat(prompts, dim=0)

                if self.is_sdxl:
                    pooled_prompts = list(
                        map(lambda x: x["pooled_embeddings"], prompts_batch)
                    )
                    pooled_prompts = torch.cat(pooled_prompts, dim=0)
            else:
                prompts = prompts_batch

            latents = list(map(lambda x: self.init_latents[x], frame_batch))
            latents = torch.cat(latents, dim=0)

            if self.video_frames is not None:
                images = list(
                    map(lambda x: self.video_frames[x].unsqueeze(0), frame_batch)
                )
                if self.video_use_pil_format:
                    images = list(map(lambda x: ToPILImage()(x[0]), images))
                else:
                    images = torch.cat(images, dim=0)

            else:
                images = []

            outputs = {
                "prompts": prompts,
                "init_latents": latents,
                "images": images,
                "frame_ids": frame_batch,
            }
            if self.is_sdxl and self.use_prompt_embeds:
                outputs.update({"pooled_prompts": pooled_prompts})

            yield outputs

    def prepare_inputs(self, batch):
        prompts = batch["prompts"]
        latents = batch["init_latents"]
        images = batch["images"]
        frame_ids = batch["frame_ids"]

        pipe_kwargs = dict(
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        )

        if "height" in self.pipe_signature:
            pipe_kwargs.update({"height": self.height})

        if "width" in self.pipe_signature:
            pipe_kwargs.update({"width": self.width})

        if "strength" in self.pipe_signature:
            pipe_kwargs.update({"strength": self.strength[frame_ids[0]]})

        if "latents" in self.pipe_signature:
            pipe_kwargs.update({"latents": latents})

        if "prompt_embeds" in self.pipe_signature and self.use_prompt_embeds:
            pipe_kwargs.update({"prompt_embeds": prompts})
            if self.is_sdxl:
                pipe_kwargs.update({"pooled_prompt_embeds": batch["pooled_prompts"]})

        elif "prompt" in self.pipe_signature and not self.use_prompt_embeds:
            pipe_kwargs.update({"prompt": prompts})

        if "negative_prompt" in self.pipe_signature:
            pipe_kwargs.update(
                {"negative_prompt": [self.negative_prompts] * len(prompts)}
            )

        if "image" in self.pipe_signature:
            if (self.video_input is not None) and (len(images) != 0):
                # preprocess the current batch of images
                image_input = self.preprocessor(images)
                pipe_kwargs.update({"image": image_input})

            elif self.image_input is not None:
                pipe_kwargs.update({"image": self.image_input})

        if "generator" in self.pipe_signature:
            pipe_kwargs.update({"generator": self.generator})

        pipe_kwargs.update(self.additional_pipeline_arguments)

        return pipe_kwargs

    def get_reference_image(self, image_input):
        if self.reference_image is None:
            self.reference_image = image_input

        return self.reference_image

    def apply_color_matching(self, image_input):
        # Color match the transformed image to the reference
        reference_image = self.get_reference_image(image_input[0])
        image_input = [
            match_color_distribution(image, reference_image) for image in image_input
        ]

        return image_input

    @torch.no_grad()
    def apply_motion(self, image, idx):
        image = image[0]
        image = image.convert("RGB")
        image_input = self.motion_callback(image, idx)

        return image_input

    def run_inference(self, batch):
        pipe_kwargs = self.prepare_inputs(batch)

        frame_id = batch["frame_ids"][0]

        if self.use_coherence:
            noise_level = self.noise_schedule[frame_id]
            self.coherence_callback.noise_level = noise_level

        output = self.diffusion_pipeline(
            **pipe_kwargs,
            callback=self.coherence_callback.apply if self.use_coherence else None,
            callback_steps=self.coherence_callback.steps if self.use_coherence else 1,
        )

        return output

    def create(self, frames=None):
        batchgen = self.batch_generator(
            frames if frames else [i for i in range(self.max_frames)], self.batch_size
        )

        for batch_idx, batch in enumerate(batchgen):
            output = self.run_inference(batch)
            images = output.images

            if self.use_color_matching_only:
                images = self.apply_color_matching(images)
                output.images = images

            yield output, batch["frame_ids"]

            if self.use_motion:
                images = self.apply_motion(images, batch)
                if self.use_color_matching:
                    images = self.apply_color_matching(images)

                self.image_input = self.preprocessor(images)
