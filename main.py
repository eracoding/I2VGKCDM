"""
main.py
Main application entry point for the image animation generation system
"""
import importlib
import os

import gradio as gr
import torch
from controlnet_aux.processor import MODELS as AUX_PROCESSORS
from wonderwords import RandomWord

from src.generate import generate_sequence
from src.session import load_session, save_session
from src.helpers import (
    get_beat_frames,
    extract_video_info,
    create_video,
    check_xformers_availability,
)

# Environment configuration
AUTO_SAVE_ENABLED = os.getenv("MODEL_AUTO_SAVE", True)
ORGANIZATION_ID = os.getenv("ORG_ID", None)
REPOSITORY_ID = os.getenv("REPO_ID", "I2VGKCDM")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
RESULTS_DIR = os.getenv("OUTPUT_BASE_PATH", "generated")
MODEL_STORAGE = os.getenv("MODEL_PATH", "models")

# Create required directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_STORAGE, exist_ok=True)

# Set up optimization
MEMORY_EFFICIENT = check_xformers_availability()
PROCESSOR_OPTIONS = ["no-processing"] + list(AUX_PROCESSORS.keys())

# Utilities
prompt_generator = gr.Interface.load("spaces/doevent/prompt-generator")
word_generator = RandomWord()


def initialize_pipeline(
    checkpoint_name, pipeline_class, ctrl_net, t2i_adapter, lora_checkpoint, custom_pipe, current_pipe
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Free up VRAM if needed
        if current_pipe is not None:
            del current_pipe
            torch.cuda.empty_cache()

        result_message = f"Successfully loaded Pipeline: {pipeline_class} with {checkpoint_name}"
        pipeline_class_obj = getattr(importlib.import_module("diffusers"), pipeline_class)

        # Check for incompatible options
        if ctrl_net and t2i_adapter:
            raise gr.Error("Cannot use both ControlNet and T2IAdapter simultaneously")

        if ctrl_net:
            from diffusers import ControlNetModel

            ctrl_net_models_list = [model.strip() for model in ctrl_net.split(",")]

            # Handle single vs multiple controlnets
            if len(ctrl_net_models_list) == 1:
                ctrl_net_models = ControlNetModel.from_pretrained(
                    ctrl_net_models_list[0], torch_dtype=torch.float16, cache_dir=MODEL_STORAGE
                )
            else:
                ctrl_net_models = [
                    ControlNetModel.from_pretrained(
                        model, torch_dtype=torch.float16, cache_dir=MODEL_STORAGE
                    )
                    for model in ctrl_net_models_list
                ]

            current_pipe = pipeline_class_obj.from_pretrained(
                checkpoint_name,
                torch_dtype=torch.float16,
                safety_checker=None,
                controlnet=ctrl_net_models,
                cache_dir=MODEL_STORAGE,
                custom_pipeline=custom_pipe if custom_pipe else None,
            )
            result_message = f"Successfully loaded Pipeline: {pipeline_class} with {checkpoint_name} and {ctrl_net_models_list}"

        elif t2i_adapter:
            from diffusers import T2IAdapter

            adapter_models_list = [model.strip() for model in t2i_adapter.split(",")]

            # Handle single vs multiple adapters
            if len(adapter_models_list) == 1:
                adapter_models = T2IAdapter.from_pretrained(
                    adapter_models_list[0], torch_dtype=torch.float16, cache_dir=MODEL_STORAGE
                )
            else:
                adapter_models = [
                    T2IAdapter.from_pretrained(
                        model, torch_dtype=torch.float16, cache_dir=MODEL_STORAGE
                    )
                    for model in adapter_models_list
                ]

            current_pipe = pipeline_class_obj.from_pretrained(
                checkpoint_name,
                torch_dtype=torch.float16,
                safety_checker=None,
                adapter=adapter_models,
                cache_dir=MODEL_STORAGE,
                custom_pipeline=custom_pipe if custom_pipe else None,
            )
            result_message = f"Successfully loaded Pipeline: {pipeline_class} with {checkpoint_name} and {adapter_models_list}"

        else:
            current_pipe = pipeline_class_obj.from_pretrained(
                checkpoint_name,
                use_auth_token=True,
                torch_dtype=torch.float16,
                safety_checker=None,
                cache_dir=MODEL_STORAGE,
                custom_pipeline=custom_pipe if custom_pipe else None,
            )

        # Apply LoRA weights if specified
        if lora_checkpoint:
            current_pipe.load_lora_weights(lora_checkpoint)

        # Optimize memory usage
        if hasattr(current_pipe, "enable_model_cpu_offload"):
            current_pipe.enable_model_cpu_offload()
        else:
            current_pipe.to(device)

        # Enable optimizations
        if hasattr(current_pipe, "enable_vae_tiling"):
            current_pipe.enable_vae_tiling()

        if MEMORY_EFFICIENT:
            current_pipe.enable_xformers_memory_efficient_attention()

        return current_pipe, result_message

    except Exception as e:
        print(e)
        return None, f"Failed to Load Pipeline: {pipeline_class} with {checkpoint_name}"


def create_prompt_sequence(frame_rate, topic_string=""):
    generated_prompts = prompt_generator(topic_string)
    prompt_lines = [
        f"{idx * frame_rate}: {prompt}" for idx, prompt in enumerate(generated_prompts.split("\n"))
    ]
    return "\n".join(prompt_lines)


def format_audio_keyframes(audio_file, frame_rate, audio_component_type):
    keyframes = get_beat_frames(audio_file, frame_rate, audio_component_type)
    return "\n".join([f"{kf}: timestamp: {kf / frame_rate:.2f}" for kf in keyframes])


def extract_video_metadata(video_file):
    total_frames, frame_rate = extract_video_info(video_file)
    return "\n".join(["0: ", f"{total_frames - 1}: "]), gr.update(value=int(frame_rate))


def pass_to_video_input(video_source):
    return video_source


def generate_output_directory():
    project_name = f"{word_generator.word(include_parts_of_speech=['adjectives'])}-{word_generator.word(include_parts_of_speech=['nouns'])}"
    output_path = os.path.join(RESULTS_DIR, project_name)
    os.makedirs(output_path, exist_ok=True)

    print(f"[DEBUG] Generated output path: {output_path}")
    return output_path


def store_session(org_id, repo_id, output_path, session_name):
    save_session(org_id, repo_id, output_path, session_name)
    display_name = session_name if session_name is not None else output_path.split("/")[-1]
    return f"Successfully saved session to {org_id}/{repo_id}/{display_name}"


def render_video(frames, output_path, frame_rate, format_type):
    print(frames)
    frame_paths = [frame.image.path for frame in frames.root]
    output_file = f"{output_path}/output.mp4"
    create_video(
        image_files=frame_paths,
        output_path=output_file,
        playback_fps=frame_rate,
    )
    return output_file


def execute_generation(
    output_path,
    pipeline,
    text_prompts,
    negative_prompts,
    img_width,
    img_height,
    steps,
    cfg_scale,
    strength_param,
    random_seed,
    batch_count,
    frame_rate,
    default_scheduler,
    sched_name,
    sched_options,
    fixed_latent,
    use_embeds,
    latent_channels,
    audio_source,
    audio_type,
    mel_reduction,
    img_source,
    video_source,
    use_pil,
    output_type,
    model_id,
    ctrl_net_id,
    adapter_id,
    lora_id,
    extra_args,
    interp_mode,
    interp_params,
    zoom_factor,
    x_translation,
    y_translation,
    rotation_angle,
    pad_mode,
    coherence_weight,
    coherence_blend,
    coherence_iters,
    noise_pattern,
    match_colors,
    preprocess_method,
):
    try:
        print("Text Prompts: ", text_prompts)
        frame_generator = generate_sequence(
            output_directory=output_path,
            pipeline=pipeline,
            prompts=text_prompts,
            negative_prompts=negative_prompts,
            img_height=int(img_height),
            img_width=int(img_width),
            inference_steps=int(steps),
            cfg_scale=cfg_scale,
            img_strength=strength_param,
            random_seed=int(random_seed),
            frames_per_batch=int(batch_count),
            frame_rate=int(frame_rate),
            custom_scheduler=default_scheduler,
            scheduler_type=sched_name,
            scheduler_options=sched_options,
            static_latent=fixed_latent,
            embed_prompts=use_embeds,
            latent_channel_count=int(latent_channels),
            audio_path=audio_source,
            audio_mode=audio_type,
            spectogram_reduction=mel_reduction,
            init_image=img_source,
            init_video=video_source,
            use_pil_video=use_pil,
            export_format=output_type,
            model_identifier=model_id,
            controlnet_identifier=ctrl_net_id,
            adapter_identifier=adapter_id,
            lora_identifier=lora_id,
            extra_pipeline_args=extra_args,
            interp_method=interp_mode,
            interp_config=interp_params,
            zoom_params=zoom_factor,
            x_shift=x_translation,
            y_shift=y_translation,
            rotation=rotation_angle,
            border_handling=pad_mode,
            temp_coherence_scale=coherence_weight,
            temp_coherence_factor=coherence_blend,
            temp_coherence_iters=int(coherence_iters),
            noise_pattern=noise_pattern,
            enable_color_matching=match_colors,
            preprocessing_method=preprocess_method,
        )

        rendered_frames = []
        for image, save_location in frame_generator:
            frame_id = save_location.split("/")[-1].split(".")[0]
            rendered_frames.append((save_location, frame_id))
            yield rendered_frames
    except Exception as e:
        raise gr.Error(e)


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌀 Text-to-Video Animation Generator
    Turn your ideas into stunning animations using Stable Diffusion and audio-visual control modules.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tab("🛠️ Settings"):
                with gr.Accordion("Pipeline Loader", open=False):
                    model_path = gr.Textbox(label="Model Name", value="runwayml/stable-diffusion-v1-5")
                    pipeline_type = gr.Textbox(label="Pipeline Type", value="DiffusionPipeline")
                    controlnet_path = gr.Textbox(label="ControlNet Checkpoint", value="")
                    adapter_path = gr.Textbox(label="T2I Adapter Checkpoint", value="")
                    lora_path = gr.Textbox(label="LoRA Checkpoint", value="")
                    custom_pipe_path = gr.Textbox(label="Custom Pipeline (Optional)", value="")
                    load_model_btn = gr.Button("🚀 Load Pipeline")
                    load_status = gr.Markdown()

                with gr.Accordion("Diffusion Settings", open=False):
                    prompt_text = gr.Textbox(label="Text Prompts", lines=4, value="0: A cat sitting on a beach\n60: A cat flying in the sky")
                    negative_text = gr.Textbox(label="Negative Prompts", lines=2, value="blurry, low quality")
                    width = gr.Number(value=512, label="Width")
                    height = gr.Number(value=512, label="Height")
                    steps = gr.Slider(10, 1000, value=20, step=10, label="Steps")
                    cfg_scale = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG Scale")
                    seed = gr.Number(value=42, label="Seed")
                    batch_size = gr.Slider(1, 32, value=1, step=1, label="Batch Size")
                    strength_param = gr.Textbox(value="0:(0.5)", label="Image Strength Schedule")

                with gr.Accordion("Animation Options", open=False):
                    frame_rate = gr.Slider(10, 60, value=24, label="Frame Rate")
                    output_format = gr.Dropdown(["mp4", "gif"], value="mp4", label="Output Format")
                    interp_mode = gr.Dropdown(["linear", "sine", "curve"], value="linear", label="Interpolation")
                    interp_params = gr.Textbox(value="", label="Interpolation Parameters")
                    zoom_factor = gr.Textbox(value="", label="Zoom Factor")
                    x_translation = gr.Textbox(value="", label="X Translation")
                    y_translation = gr.Textbox(value="", label="Y Translation")
                    rotation_angle = gr.Textbox(value="", label="Rotation Angle")
                    pad_mode = gr.Dropdown(["zero", "border", "reflection"], value="border", label="Padding Mode")
                    coherence_weight = gr.Slider(0, 100000, value=0, step=50, label="Coherence Weight")
                    coherence_blend = gr.Slider(0, 1.0, value=1.0, step=0.1, label="Coherence Blend")
                    coherence_iters = gr.Slider(0, 100, value=1, step=1, label="Coherence Iterations")
                    noise_pattern = gr.Textbox(value="0:(0.01)", label="Noise Pattern")
                    match_colors = gr.Checkbox(value=False, label="Match Colors")

                    default_scheduler = gr.Checkbox(value=False, label="Use Default Scheduler")
                    sched_name = gr.Dropdown(
                        ["klms", "ddim", "ddpm", "pndms", "dpm", "dpm_ads", "deis", "euler", "euler_ads", "unipc"],
                        value="deis",
                        label="Scheduler"
                    )
                    sched_options = gr.Textbox(value="{}", label="Scheduler Options")
                    fixed_latent = gr.Checkbox(value=False, label="Fixed Latent")
                    use_embeds = gr.Checkbox(value=False, label="Use Embeds")
                    latent_channels = gr.Number(value=4, label="Latent Channels")

            with gr.Tab("🎨 Media Inputs"):
                image_input = gr.Image(label="Image Input", type="pil")
                video_input = gr.Video(label="Video Input")
                audio_input = gr.Audio(label="Audio Input", type="filepath")

                audio_type = gr.Dropdown(["percussive", "harmonic", "both"], value="percussive", label="Audio Type")
                mel_reduce = gr.Dropdown(["mean", "median", "max"], value="max", label="Mel Reduction")
                preprocess = gr.Dropdown(["no-processing"], value=["no-processing"], label="Preprocessing", multiselect=True)

        with gr.Column(scale=3):
            with gr.Tab("🖼️ Outputs"):
                gallery = gr.Gallery(label="Generated Frames", columns=4)
                video_display = gr.Video(label="Rendered Video")

                with gr.Row():
                    create_btn = gr.Button("🎬 Generate Animation")
                    stop_btn = gr.Button("🛑 Stop")
                    render_btn = gr.Button("💾 Render to Video")

    # State and Callbacks
    model_state = gr.State()
    output_state = gr.State()

    load_model_btn.click(
        fn=initialize_pipeline, 
        inputs=[model_path, pipeline_type, controlnet_path, adapter_path, lora_path, custom_pipe_path, model_state],
        outputs=[model_state, load_status]
    )

    generate_event = create_btn.click(fn=generate_output_directory, outputs=[output_state])

    generate_event.then(
        fn=execute_generation,
        inputs=[
            output_state, model_state, prompt_text, negative_text, width, height, steps, cfg_scale,
            strength_param, seed, batch_size, frame_rate, default_scheduler, sched_name, sched_options,
            fixed_latent, use_embeds, latent_channels, audio_input, audio_type, mel_reduce, image_input,
            video_input, gr.Checkbox(value=True, label="use_pil"), output_format, model_path, controlnet_path, adapter_path,
            lora_path, gr.Textbox(value="{}", label="extra_args"), interp_mode, interp_params, zoom_factor, x_translation,
            y_translation, rotation_angle, pad_mode, coherence_weight, coherence_blend, coherence_iters,
            noise_pattern, match_colors, preprocess
        ],
        outputs=[gallery]
    )

    render_btn.click(
        fn=render_video,
        inputs=[gallery, output_state, frame_rate, output_format],
        outputs=[video_display]
    )


if __name__ == "__main__":
    demo.launch(share=True, debug=DEBUG_MODE)
