import streamlit as st
import os
import torch
from PIL import Image
from pathlib import Path
import tempfile
import time
from src.generate import generate_sequence
from src.pipelines import custom_pipeline

# Set page config
st.set_page_config(
    page_title="Giffusion Web App",
    page_icon="🎞️",
    layout="wide",
)

# Function to create a download link for files
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    file_name = os.path.basename(bin_file)
    return f'<a href="data:application/octet-stream;base64,{data}" download="{file_name}">{file_label}</a>'

# App title and description
st.title("I2VGKCDM: Text-to-Video Generator")
st.markdown("""Generate smooth video animations from text prompts using diffusion models.""")

# Sidebar for model selection and general settings
st.sidebar.header("Model Settings")

# Model selection
model_type = st.sidebar.radio(
    "Generation Mode",
    ["Standard (Single Prompt)", "Advanced (Prompt Transitions)"]
)

model_id = st.sidebar.selectbox(
    "Diffusion Model",
    ["CompVis/stable-diffusion-v1-4", "runwayml/stable-diffusion-v1-5", "stabilityai/stable-diffusion-2-1"]
)

use_gpu = st.sidebar.checkbox("Use GPU (if available)", value=True)
device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

seed = st.sidebar.number_input("Random Seed (leave at 0 for random)", min_value=0, value=0)
if seed == 0:
    seed = None

# General settings
height = st.sidebar.select_slider("Image Height", options=[256, 320, 384, 448, 512, 576, 640, 704, 768], value=512)
width = st.sidebar.select_slider("Image Width", options=[256, 320, 384, 448, 512, 576, 640, 704, 768], value=512)
fps = st.sidebar.slider("Frames Per Second", min_value=1, max_value=30, value=8)

# Main area for prompt input and generation
if model_type == "Standard (Single Prompt)":
    st.header("Generate Video from Single Prompt")
    
    prompt = st.text_area("Enter your prompt", "A beautiful landscape with mountains and lakes, trending on artstation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_frames = st.slider("Number of Frames", min_value=8, max_value=48, value=16)
    
    with col2:
        variation_strength = st.slider("Variation Strength", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    
    guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    
    gen_button = st.button("Generate Video")
    
    if gen_button:
        if not prompt:
            st.error("Please enter a prompt")
        else:
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    with st.spinner("Initializing generator..."):
                        generator = generate_sequence(model_id=model_id, device=device)
                    
                    start_time = time.time()
                    with st.spinner(f"Generating {n_frames} frames from prompt..."):
                        images = generator.generate_image_sequence(
                            prompt=prompt,
                            n_frames=n_frames,
                            variation_strength=variation_strength,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            height=height,
                            width=width
                        )
                    
                    # Create a progress display
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Save individual frames and progress display
                    frames_dir = os.path.join(temp_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    for i, img in enumerate(images):
                        img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
                        progress = (i + 1) / len(images)
                        progress_bar.progress(progress)
                        status_text.text(f"Saving frame {i+1}/{len(images)}")
                    
                    # Create video
                    status_text.text("Creating video...")
                    video_path = os.path.join(temp_dir, "video.mp4")
                    generator.create_video(images, output_path=video_path, fps=fps)
                    
                    # Create GIF
                    status_text.text("Creating GIF...")
                    gif_path = os.path.join(temp_dir, "animation.gif")
                    generator.create_gif(images, output_path=gif_path, duration=int(1000/fps))
                    
                    # Display results
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    status_text.text(f"Generation completed in {generation_time:.2f} seconds")
                    progress_bar.empty()
                    
                    # Display results
                    st.subheader("Generated Results")
                    
                    # Display the video
                    st.video(video_path)
                    
                    # Display first and last frames
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(images[0], caption="First Frame")
                    with col2:
                        st.image(images[-1], caption="Last Frame")
                    
                    # Provide download links
                    st.subheader("Download Files")
                    st.markdown(f"Right-click and 'Save link as...' to download")
                    
                    video_file = open(video_path, "rb")
                    st.download_button(
                        label="Download Video",
                        data=video_file,
                        file_name="giffusion_video.mp4",
                        mime="video/mp4"
                    )
                    
                    gif_file = open(gif_path, "rb")
                    st.download_button(
                        label="Download GIF",
                        data=gif_file,
                        file_name="giffusion_animation.gif",
                        mime="image/gif"
                    )
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

else:  # Advanced mode
    st.header("Generate Video with Prompt Transitions")
    
    st.markdown("Enter multiple prompts to create smooth transitions between them")
    
    # Create a list for multiple prompts
    if 'prompts' not in st.session_state:
        st.session_state.prompts = ["A sunny beach with palm trees", "A snowy mountain landscape"]
    
    # Function to add a new prompt
    def add_prompt():
        st.session_state.prompts.append("")
    
    # Function to remove a prompt
    def remove_prompt(i):
        st.session_state.prompts.pop(i)
    
    # Display and edit prompts
    for i, prompt in enumerate(st.session_state.prompts):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.session_state.prompts[i] = st.text_input(f"Prompt {i+1}", prompt, key=f"prompt_{i}")
        with col2:
            if len(st.session_state.prompts) > 2:  # Don't allow removing if only 2 prompts left
                st.button(f"Remove", key=f"remove_{i}", on_click=remove_prompt, args=(i,))
    
    st.button("Add Another Prompt", on_click=add_prompt)
    
    # Settings
    col1, col2 = st.columns(2)
    
    with col1:
        frames_per_transition = st.slider("Frames Per Transition", min_value=4, max_value=24, value=8)
    
    with col2:
        guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=7.5, step=0.5)
    
    gen_button = st.button("Generate Transition Video")
    
    if gen_button:
        # Check if prompts are valid
        if len(st.session_state.prompts) < 2 or any(not p.strip() for p in st.session_state.prompts):
            st.error("Please enter at least two valid prompts")
        else:
            try:
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    with st.spinner("Initializing advanced generator..."):
                        generator = generate_sequence(model_id=model_id, device=device)
                    
                    start_time = time.time()
                    with st.spinner(f"Generating transitions between {len(st.session_state.prompts)} prompts..."):
                        images = generator.generate_multi_prompt_sequence(
                            prompts=st.session_state.prompts,
                            frames_per_transition=frames_per_transition,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            height=height,
                            width=width
                        )
                    
                    # Create a progress display
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Save individual frames with progress display
                    frames_dir = os.path.join(temp_dir, "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    
                    for i, img in enumerate(images):
                        img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
                        progress = (i + 1) / len(images)
                        progress_bar.progress(progress)
                        status_text.text(f"Saving frame {i+1}/{len(images)}")
                    
                    # Create video
                    status_text.text("Creating video...")
                    video_path = os.path.join(temp_dir, "video.mp4")
                    generator.create_video(images, output_path=video_path, fps=fps)
                    
                    # Create GIF
                    status_text.text("Creating GIF...")
                    gif_path = os.path.join(temp_dir, "animation.gif")
                    generator.create_gif(images, output_path=gif_path, duration=int(1000/fps))
                    
                    # Display results
                    end_time = time.time()
                    generation_time = end_time - start_time
                    
                    status_text.text(f"Generation completed in {generation_time:.2f} seconds")
                    progress_bar.empty()
                    
                    # Display results
                    st.subheader("Generated Results")
                    
                    # Display the video
                    st.video(video_path)
                    
                    # Display samples of frames
                    st.subheader("Sample Frames")
                    cols = st.columns(len(st.session_state.prompts))
                    
                    # Calculate indices for frames that represent each prompt
                    indices = [int(i * len(images) / (len(st.session_state.prompts))) for i in range(len(st.session_state.prompts))]
                    
                    for i, col in enumerate(cols):
                        with col:
                            st.image(images[indices[i]], caption=f"Prompt {i+1}")
                    
                    # Provide download links
                    st.subheader("Download Files")
                    st.markdown(f"Right-click and 'Save link as...' to download")
                    
                    video_file = open(video_path, "rb")
                    st.download_button(
                        label="Download Video",
                        data=video_file,
                        file_name="giffusion_transition_video.mp4",
                        mime="video/mp4"
                    )
                    
                    gif_file = open(gif_path, "rb")
                    st.download_button(
                        label="Download GIF",
                        data=gif_file,
                        file_name="giffusion_transition_animation.gif",
                        mime="image/gif"
                    )
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("Giffusion: Text-to-Video Generator | Course Project Implementation")
