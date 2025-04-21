"""
---------------------------------
File: generate.py
Author: Ulugbek Shernazarov
Email: u.shernaz4rov@gmail.com
Copyright (c) 2025 Ulugbek Shernazarov. All rights reserved | GitHub: eracoding
Description: Core module for generating image sequences using diffusion models. Handles configuration of scheduler, pipeline initialization, and frame generation process with various parameters for animation effects, motion control, and temporal coherence.
---------------------------------
"""
import json
import logging
import os
from datetime import datetime

import typer
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    RePaintScheduler,
    UniPCMultistepScheduler,
)
from diffusers.utils.logging import disable_progress_bar
from tqdm import tqdm

from src.pipelines.custom_pipeline import CustomPipeline
from src.helpers import store_config

# Configure logging and disable denoising progress display
log_manager = logging.getLogger(__name__)
disable_progress_bar()


def get_scheduler_instance(scheduler_type, **config):
    """Create and return appropriate scheduler based on specified type"""
    scheduler_registry = {
        "pndms": PNDMScheduler,
        "ddim": DDIMScheduler,
        "ddpm": DDPMScheduler,
        "klms": LMSDiscreteScheduler,
        "dpm": DPMSolverSinglestepScheduler,
        "dpm_ads": KDPM2AncestralDiscreteScheduler,
        "deis": DEISMultistepScheduler,
        "euler": EulerDiscreteScheduler,
        "euler_ads": EulerAncestralDiscreteScheduler,
        "repaint": RePaintScheduler,
        "unipc": UniPCMultistepScheduler,
    }
    
    if scheduler_type in scheduler_registry:
        return scheduler_registry[scheduler_type](**config)
    return None


def generate_sequence(
    output_directory,
    pipeline,
    prompts,
    negative_prompts,
    img_height=512,
    img_width=512,
    inference_steps=50,
    cfg_scale=7.5,
    img_strength=0.5,
    frames_per_batch=1,
    random_seed=42,
    frame_rate=24,
    custom_scheduler=False,
    scheduler_type="pndms",
    scheduler_options="{}",
    static_latent=False,
    embed_prompts=True,
    latent_channel_count=4,
    audio_path=None,
    audio_mode="both",
    spectogram_reduction="max",
    init_image=None,
    init_video=None,
    use_pil_video=False,
    export_format="mp4",
    model_identifier="runwayml/stable-diffusion-v1-5",
    controlnet_identifier=None,
    adapter_identifier=None,
    lora_identifier=None,
    extra_pipeline_args="{}",
    interp_method="linear",
    interp_config="",
    zoom_params="",
    x_shift="",
    y_shift="",
    rotation="",
    border_handling="border",
    temp_coherence_scale=300,
    temp_coherence_factor=1.0,
    temp_coherence_iters=3,
    noise_pattern=None,
    enable_color_matching=False,
    preprocessing_method=None,
):
    """
    Generate a sequence of images using diffusion models with various controls and parameters.
    """
    if pipeline is None:
        raise ValueError("Pipeline must be initialized before generating sequences")

    # Create timestamp and identify target device
    current_time = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    target_device = pipeline.device

    # Configure custom scheduler if requested
    if not custom_scheduler:
        scheduler_config = json.loads(scheduler_options)
        if not scheduler_config:
            scheduler_config = {
                "beta_start": 0.00085,
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
            }
        pipeline.scheduler = get_scheduler_instance(scheduler_type, **scheduler_config)

    # Setup motion parameters
    motion_config = {
        "zoom": zoom_params,
        "translate_x": x_shift,
        "translate_y": y_shift,
        "angle": rotation,
    }

    # Parse additional pipeline arguments
    extra_args = json.loads(extra_pipeline_args)

    # Initialize the generation flow
    generation_flow = CustomPipeline(
        diffusion_pipeline=pipeline,
        text_conditioning=prompts,
        negative_conditioning=negative_prompts,
        guidance_scale=cfg_scale,
        strength_curve=img_strength,
        inference_steps=inference_steps,
        image_height=img_height,
        image_width=img_width,
        use_consistent_latent=static_latent,
        use_text_embeddings=embed_prompts,
        latent_channel_count=latent_channel_count,
        compute_device=target_device,
        input_image=init_image,
        audio_source=audio_path,
        audio_type=audio_mode,
        audio_reduction_method=spectogram_reduction,
        video_source=init_video,
        use_pil_video_format=use_pil_video,
        random_seed=random_seed,
        batch_size=frames_per_batch,
        frames_per_second=frame_rate,
        additional_pipeline_params=extra_args,
        interpolation_mode=interp_method,
        interpolation_settings=interp_config,
        transform_settings=motion_config,
        boundary_handling=border_handling,
        stability_strength=temp_coherence_scale,
        memory_persistence=temp_coherence_factor,
        stability_iterations=temp_coherence_iters,
        noise_curve=noise_pattern,
        enable_color_matching=enable_color_matching,
        image_processors=preprocessing_method,
    )

    # Get total frame count and prepare config for saving
    total_frames = generation_flow.total_frames
    config_data = {
        "prompts": {
            "text_prompt_inputs": prompts,
            "negative_prompt_inputs": negative_prompts,
        },
        "diffusion_settings": {
            "num_inference_steps": inference_steps,
            "guidance_scale": cfg_scale,
            "strength": img_strength,
            "batch_size": frames_per_batch,
            "seed": random_seed,
            "use_fixed_latent": static_latent,
            "use_prompt_embeds": embed_prompts,
            "scheduler": scheduler_type,
            "use_default_scheduler": custom_scheduler,
            "scheduler_kwargs": scheduler_config,
            "image_height": img_height,
            "image_width": img_width,
            "additional_pipeline_arguments": extra_args,
        },
        "preprocessing_settings": {
            "preprocess": preprocessing_method,
        },
        "pipeline_settings": {
            "pipeline_name": pipeline.__class__.__name__,
            "model_name": model_identifier,
            "controlnet_name": controlnet_identifier,
            "adapter_name": adapter_identifier,
            "lora_name": lora_identifier,
        },
        "animation_settings": {
            "interpolation_type": interp_method,
            "interpolation_args": interp_config,
            "zoom": zoom_params,
            "translate_x": x_shift,
            "translate_y": y_shift,
            "angle": rotation,
            "padding_mode": border_handling,
            "coherence_scale": temp_coherence_scale,
            "coherence_alpha": temp_coherence_factor,
            "coherence_steps": temp_coherence_iters,
            "noise_schedule": noise_pattern,
            "use_color_matching": enable_color_matching,
        },
        "media": {
            "audio_settings": {
                "audio_component": audio_mode,
                "mel_spectogram_reduce": spectogram_reduction,
            },
            "video_settings": {
                "video_use_pil_format": use_pil_video,
            },
        },
        "output_settings": {
            "output_format": export_format,
            "fps": frame_rate,
        },
        "frame_information": {"last_frame_id": total_frames},
        "timestamp": current_time,
    }
    
    # Add strength parameter when using image or video input
    if (init_video is not None) or (init_image is not None):
        config_data.update({"strength": img_strength})

    # Save parameters to output directory
    print("Output directory: ", output_directory)
    store_config(output_directory, config_data)

    # Create directory for image outputs
    images_dir = f"{output_directory}/imgs"
    os.makedirs(images_dir, exist_ok=True)

    # Generate and save images
    frame_generator = generation_flow.generate_frames()
    for output_batch, frame_indices in tqdm(frame_generator, total=total_frames // generation_flow.batch_size):
        rendered_images = output_batch.images
        for frame_img, frame_idx in zip(rendered_images, frame_indices):
            output_path = f"{images_dir}/{frame_idx:04d}.png"
            frame_img.save(output_path)
            yield frame_img, output_path


if __name__ == "__main__":
    typer.run(generate_sequence)
