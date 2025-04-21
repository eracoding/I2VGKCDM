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
AUTO_SAVE_ENABLED = os.getenv("GIFFUSION_AUTO_SAVE", True)
ORGANIZATION_ID = os.getenv("ORG_ID", None)
REPOSITORY_ID = os.getenv("REPO_ID", "giffusion")
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
    return output_path


def store_session(org_id, repo_id, output_path, session_name):
    save_session(org_id, repo_id, output_path, session_name)
    display_name = session_name if session_name is not None else output_path.split("/")[-1]
    return f"Successfully saved session to {org_id}/{repo_id}/{display_name}"


def render_video(frames, output_path, frame_rate, format_type):
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


# UI Definition
interface = gr.Blocks()

with interface:
    gr.Markdown("# Animation Generation")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Session Settings", open=False):
                with gr.Tab("Save"):
                    org_id_input = gr.Textbox(label="Org ID", value=ORGANIZATION_ID)
                    repo_id_input = gr.Textbox(label="Repo ID", value=REPOSITORY_ID)
                    session_name_input = gr.Textbox(label="Session Name")
                    save_btn = gr.Button(value="Save Session")
                    save_status = gr.Markdown()

                with gr.Tab("Load"):
                    load_org_input = gr.Textbox(label="Org ID", value=ORGANIZATION_ID)
                    load_repo_input = gr.Textbox(label="Repo ID", value=REPOSITORY_ID)
                    load_session_input = gr.Textbox(label="Session Name")
                    settings_filter = gr.Dropdown(
                        [
                            "prompts",
                            "diffusion_settings",
                            "preprocessing_settings",
                            "pipeline_settings",
                            "animation_settings",
                        ],
                        label="Filter Settings",
                        multiselect=True,
                    )
                    load_btn = gr.Button(value="Load Session Settings")

            with gr.Accordion("Pipeline Settings: Load Models and Pipelines"):
                with gr.Column():
                    model_path = gr.Textbox(
                        label="Model Name", value="runwayml/stable-diffusion-v1-5"
                    )
                    pipeline_type = gr.Textbox(
                        label="Pipeline Name", value="DiffusionPipeline"
                    )
                    lora_path = gr.Textbox(label="LoRA Checkpoint")

                    with gr.Tab("ControlNet"):
                        controlnet_path = gr.Textbox(label="ControlNet Checkpoint")
                    with gr.Tab("T2I Adapters"):
                        adapter_path = gr.Textbox(label="T2I Adapter Checkpoint")

                    custom_pipe_path = gr.Textbox(label="Custom Pipeline")

                with gr.Column():
                    with gr.Row():
                        load_model_btn = gr.Button(value="Load Pipeline")
                    with gr.Row():
                        load_status_msg = gr.Markdown()

            with gr.Accordion(
                "Output Settings: Set output file format and FPS", open=False
            ):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            output_format_selector = gr.Dropdown(
                                ["gif", "mp4"], value="mp4", label="Output Format"
                            )
                        with gr.Row():
                            framerate_slider = gr.Slider(
                                10, 60, step=1, value=10, label="Output Frame Rate"
                            )
                        with gr.Row():
                            render_btn = gr.Button(value="Save Video")

            with gr.Accordion("Diffusion Settings", open=False):
                with gr.Tab("Diffusion"):
                    fixed_latent_toggle = gr.Checkbox(
                        label="Use Fixed Init Latent", elem_id="use_fixed_latent"
                    )
                    prompt_embed_toggle = gr.Checkbox(
                        label="Use Prompt Embeds",
                        value=False,
                        interactive=True,
                        elem_id="use_prompt_embed",
                    )
                    seed_number = gr.Number(value=42, label="Numerical Seed", elem_id="seed")
                    batch_slider = gr.Slider(
                        1, 64, step=1, value=1, label="Batch Size", elem_id="batch_size"
                    )
                    steps_slider = gr.Slider(
                        10,
                        1000,
                        step=10,
                        value=20,
                        label="Number of Iteration Steps",
                        elem_id="num_iteration_steps",
                    )
                    cfg_slider = gr.Slider(
                        0.5,
                        20,
                        step=0.5,
                        value=7.5,
                        label="Classifier Free Guidance Scale",
                        elem_id="guidance_scale",
                    )
                    strength_input = gr.Textbox(
                        label="Image Strength Schedule",
                        value="0:(0.5)",
                        elem_id="strength",
                    )
                    latent_channel_count = gr.Number(
                        value=4,
                        label="Number of Latent Channels",
                        elem_id="num_latent_channels",
                    )
                    height_input = gr.Number(
                        value=512, label="Image Height", elem_id="image_height"
                    )
                    width_input = gr.Number(
                        value=512, label="Image Width", elem_id="image_width"
                    )

                with gr.Tab("Scheduler"):
                    default_sched_toggle = gr.Checkbox(
                        label="Use Default Pipeline Scheduler",
                        elem_id="use_default_scheduler",
                    )
                    scheduler_type = gr.Dropdown(
                        [
                            "klms",
                            "ddim",
                            "ddpm",
                            "pndms",
                            "dpm",
                            "dpm_ads",
                            "deis",
                            "euler",
                            "euler_ads",
                            "unipc",
                        ],
                        value="deis",
                        label="Scheduler",
                        elem_id="scheduler",
                    )
                    scheduler_params = gr.Textbox(
                        label="Scheduler Arguments",
                        value="{}",
                        elem_id="scheduler_kwargs",
                    )

                with gr.Tab("Pipeline"):
                    extra_pipeline_args = gr.Textbox(
                        label="Additional Pipeline Arguments",
                        value="{}",
                        interactive=True,
                        lines=4,
                        placeholder="A dictionary of key word arguments to pass to the pipeline",
                        elem_id="additional_pipeline_arguments",
                    )

            with gr.Accordion("Animation Settings", open=False):
                with gr.Tab("Interpolation"):
                    interp_selector = gr.Dropdown(
                        ["linear", "sine", "curve"],
                        value="linear",
                        label="Interpolation Type",
                        elem_id="interpolation_type",
                    )
                    interp_params_input = gr.Textbox(
                        "",
                        label="Interpolation Parameters",
                        visible=True,
                        elem_id="interpolation_args",
                    )
                with gr.Tab("Motion"):
                    zoom_input = gr.Textbox("", label="Zoom", elem_id="zoom")
                    translate_x_input = gr.Textbox(
                        "", label="Translate_X", elem_id="translate_x"
                    )
                    translate_y_input = gr.Textbox(
                        "", label="Translate_Y", elem_id="translate_y"
                    )
                    angle_input = gr.Textbox("", label="Angle", elem_id="angle")
                    padding_selector = gr.Dropdown(
                        ["zero", "border", "reflection"],
                        label="Padding Mode",
                        value="border",
                        elem_id="padding_mode",
                    )

                with gr.Tab("Coherence"):
                    coherence_slider = gr.Slider(
                        0,
                        100000,
                        step=50,
                        value=0,
                        label="Coherence Scale",
                        elem_id="coherence",
                    )
                    coherence_alpha_slider = gr.Slider(
                        0,
                        1.0,
                        step=0.1,
                        value=1.0,
                        label="Coherence Alpha",
                        elem_id="coherence_alpha",
                    )
                    coherence_steps_slider = gr.Slider(
                        0,
                        100,
                        step=1,
                        value=1,
                        label="Coherence Steps",
                        elem_id="coherence_steps",
                    )
                    noise_sched_input = gr.Textbox(
                        label="Noise Schedule",
                        value="0:(0.01)",
                        interactive=True,
                        elem_id="noise_schedule",
                    )
                    color_match_toggle = gr.Checkbox(
                        label="Apply Color Matching",
                        value=False,
                        interactive=True,
                        elem_id="use_color_matching",
                    )

            with gr.Accordion("Inspiration Settings", open=False):
                with gr.Row():
                    topic_input = gr.Textbox(lines=1, value="", label="Inspiration Topics")

                with gr.Row():
                    inspire_btn = gr.Button(
                        value="Give me some inspiration!",
                        variant="secondary",
                        elem_id="prompt-generator-btn",
                    )

        with gr.Column(elem_id="output", scale=2):
            with gr.Row():
                with gr.Tab("Output"):
                    gallery = gr.Gallery(
                        label="Current Generation",
                        preview=True,
                        elem_id="preview",
                        show_label=True,
                    )
                    send_to_img_btn = gr.Button(value="Send to Image Input")

                with gr.Tab("Video Output"):
                    video_output = gr.Video(label="Model Output", elem_id="output")
                    send_to_vid_btn = gr.Button(value="Send to Video Input")

            with gr.Row():
                create_btn = gr.Button(
                    value="Create",
                    variant="primary",
                    elem_id="submit-btn",
                )
                cancel_btn = gr.Button(
                    value="Stop",
                    elem_id="stop-btn",
                )
            with gr.Row():
                prompt_text = gr.Textbox(
                    lines=10,
                    value="""0: A corgi in the clouds\n60: A corgi in the ocean""",
                    label="Text Prompts",
                    interactive=True,
                    elem_id="text_prompt_inputs",
                )
            with gr.Row():
                negative_text = gr.Textbox(
                    value="""low resolution, blurry, worst quality, jpeg artifacts""",
                    label="Negative Prompts",
                    interactive=True,
                    elem_id="negative_prompt_inputs",
                )

        with gr.Column(scale=1):
            with gr.Accordion("Image Input", open=False):
                img_source_input = gr.Image(label="Initial Image", type="pil")

            with gr.Accordion("Audio Input", open=False):
                audio_source_input = gr.Audio(label="Audio Input", type="filepath")
                audio_type_selector = gr.Dropdown(
                    ["percussive", "harmonic", "both"],
                    value="percussive",
                    label="Audio Component",
                    elem_id="audio_component",
                )
                audio_keyframe_btn = gr.Button(value="Get Key Frame Information")
                mel_reduction_method = gr.Dropdown(
                    ["mean", "median", "max"],
                    label="Mel Spectrogram Reduction",
                    value="max",
                    elem_id="mel_spectrogram_reduce",
                )

            with gr.Accordion("Video Input", open=False):
                video_source_input = gr.Video(label="Video Input")
                video_keyframe_btn = gr.Button(value="Get Key Frame Infomation")
                pil_format_toggle = gr.Checkbox(label="Use PIL Format", value=True)

            with gr.Accordion("Controlnet Preprocessing Settings", open=False):
                preprocess_selector = gr.Dropdown(
                    PROCESSOR_OPTIONS,
                    label="Preprocessing",
                    multiselect=True,
                    elem_id="preprocess",
                )

    # State variables
    model_pipeline = gr.State()
    output_dir = gr.State()
    selected_frame = gr.State()

    # Button handlers
    load_model_btn.click(
        initialize_pipeline,
        [model_path, pipeline_type, controlnet_path, adapter_path, lora_path, custom_pipe_path, model_pipeline],
        [model_pipeline, load_status_msg],
    )

    inspire_btn.click(
        create_prompt_sequence,
        inputs=[framerate_slider, topic_input],
        outputs=prompt_text,
    )
    
    audio_keyframe_btn.click(
        format_audio_keyframes,
        inputs=[audio_source_input, framerate_slider, audio_type_selector],
        outputs=[prompt_text],
    )
    
    video_keyframe_btn.click(
        extract_video_metadata,
        inputs=[video_source_input],
        outputs=[prompt_text, framerate_slider],
    )

    init_event = create_btn.click(generate_output_directory, outputs=[output_dir])
    generation_event = init_event.success(
        fn=execute_generation,
        inputs=[
            output_dir,
            model_pipeline,
            prompt_text,
            negative_text,
            width_input,
            height_input,
            steps_slider,
            cfg_slider,
            strength_input,
            seed_number,
            batch_slider,
            framerate_slider,
            default_sched_toggle,
            scheduler_type,
            scheduler_params,
            fixed_latent_toggle,
            prompt_embed_toggle,
            latent_channel_count,
            audio_source_input,
            audio_type_selector,
            mel_reduction_method,
            img_source_input,
            video_source_input,
            pil_format_toggle,
            output_format_selector,
            model_path,
            controlnet_path,
            adapter_path,
            lora_path,
            extra_pipeline_args,
            interp_selector,
            interp_params_input,
            zoom_input,
            translate_x_input,
            translate_y_input,
            angle_input,
            padding_selector,
            coherence_slider,
            coherence_alpha_slider,
            coherence_steps_slider,
            noise_sched_input,
            color_match_toggle,
            preprocess_selector,
        ],
        outputs=[gallery],
    )

    cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[generation_event])
    save_btn.click(
        store_session,
        inputs=[org_id_input, repo_id_input, output_dir, session_name_input],
        outputs=[save_status],
    )

    def handle_select(evt: gr.SelectData):
        item = evt.value
        img_path = item["image"]["path"]
        return img_path

    gallery.select(handle_select, None, outputs=[selected_frame])

    def pass_to_image_input(frame):
        return frame

    send_to_img_btn.click(pass_to_image_input, selected_frame, img_source_input)

    def restore_session(org_id, repo_id, session_name, settings_filter):
        config = load_session(org_id, repo_id, session_name)
        if settings_filter:
            filtered_config = {k: config[k] for k in settings_filter}
            config = filtered_config

        result = {}
        for section, settings in config.items():
            if section == "pipeline_settings":
                result.update({
                    model_path: config["pipeline_settings"]["model_name"],
                    pipeline_type: config["pipeline_settings"]["pipeline_name"],
                    lora_path: config["pipeline_settings"]["lora_name"],
                    controlnet_path: config["pipeline_settings"]["controlnet_name"],
                    adapter_path: config["pipeline_settings"]["adapter_name"],
                })

            if section == "prompts":
                result.update({
                    prompt_text: config["prompts"]["text_prompt_inputs"],
                    negative_text: config["prompts"]["negative_prompt_inputs"],
                })

            if section == "diffusion_settings":
                result.update({
                    height_input: config["diffusion_settings"]["image_height"],
                    width_input: config["diffusion_settings"]["image_width"],
                    steps_slider: config["diffusion_settings"]["num_inference_steps"],
                    cfg_slider: config["diffusion_settings"]["guidance_scale"],
                    strength_input: config["diffusion_settings"]["strength"],
                    seed_number: config["diffusion_settings"]["seed"],
                    batch_slider: config["diffusion_settings"]["batch_size"],
                    scheduler_type: config["diffusion_settings"]["scheduler"],
                    default_sched_toggle: config["diffusion_settings"]["use_default_scheduler"],
                    prompt_embed_toggle: config["diffusion_settings"]["use_prompt_embeds"],
                    fixed_latent_toggle: config["diffusion_settings"]["use_fixed_latent"],
                    extra_pipeline_args: config["diffusion_settings"]["additional_pipeline_arguments"],
                })
                
            if section == "animation_settings":
                result.update({
                    interp_selector: config["animation_settings"]["interpolation_type"],
                    interp_params_input: config["animation_settings"]["interpolation_args"],
                    zoom_input: config["animation_settings"]["zoom"],
                    translate_x_input: config["animation_settings"]["translate_x"],
                    translate_y_input: config["animation_settings"]["translate_y"],
                    angle_input: config["animation_settings"]["angle"],
                    padding_selector: config["animation_settings"]["padding_mode"],
                    coherence_slider: config["animation_settings"]["coherence_scale"],
                    coherence_alpha_slider: config["animation_settings"]["coherence_alpha"],
                    coherence_steps_slider: config["animation_settings"]["coherence_steps"],
                    noise_sched_input: config["animation_settings"]["noise_schedule"],
                    color_match_toggle: config["animation_settings"]["use_color_matching"],
                })
                
            if section == "preprocessing_settings":
                result.update({
                    preprocess_selector: config["preprocessing_settings"]["preprocess"]
                })

        return result

    load_btn.click(
        restore_session,
        [
            load_org_input,
            load_repo_input,
            load_session_input,
            settings_filter,
        ],
        outputs=[
            pipeline_type,
            model_path,
            lora_path,
            controlnet_path,
            adapter_path,
            custom_pipe_path,
            prompt_text,
            negative_text,
            width_input,
            height_input,
            steps_slider,
            cfg_slider,
            strength_input,
            seed_number,
            batch_slider,
            framerate_slider,
            default_sched_toggle,
            scheduler_type,
            scheduler_params,
            fixed_latent_toggle,
            prompt_embed_toggle,
            latent_channel_count,
            audio_source_input,
            audio_type_selector,
            mel_reduction_method,
            img_source_input,
            video_source_input,
            pil_format_toggle,
            output_format_selector,
            model_path,
            controlnet_path,
            extra_pipeline_args,
            interp_selector,
            interp_params_input,
            zoom_input,
            translate_x_input,
            translate_y_input,
            angle_input,
            padding_selector,
            coherence_slider,
            coherence_alpha_slider,
            coherence_steps_slider,
            noise_sched_input,
            color_match_toggle,
            preprocess_selector,
        ],
    )
    
    render_btn.click(render_video, [gallery, output_dir, framerate_slider, output_format_selector], video_output)
    send_to_vid_btn.click(pass_to_video_input, video_output, video_source_input)

if __name__ == "__main__":
    interface.launch(share=True, debug=DEBUG_MODE)
