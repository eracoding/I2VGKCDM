"""
File: helpers.py
"""
import json
import re
from typing import List, Dict, Any, Callable, Union, Tuple
import torch
import numpy as np
import librosa
from PIL import Image
from keyframed.dsl import curve_from_cn_string
from kornia.color import lab_to_rgb, rgb_to_lab
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
from skimage.exposure import match_histograms
from torchvision.io import read_video, write_video
from torchvision.transforms import ToPILImage, ToTensor


def execute_spatial_transform(
    tensor_img, transform_params, edge_handling="border", background_value=torch.zeros(3)
):
    """Apply spatial transformations to an image tensor"""
    batch_count, channels, height, width = tensor_img.shape
    image_center = torch.tensor((height / 2, width / 2)).unsqueeze(0)
    # Extract transform parameters
    scaling = torch.tensor([transform_params["scale"], transform_params["scale"]]).unsqueeze(0)
    horizontal_shift = transform_params["horizontal_shift"]
    vertical_shift = transform_params["vertical_shift"]
    shift = torch.tensor((horizontal_shift, vertical_shift)).unsqueeze(0)
    rotation = torch.tensor([transform_params["rotation"]])

    # Generate transform matrix
    transform_matrix = get_affine_matrix2d(
        center=image_center, translations=shift, angle=rotation, scale=scaling
    )

    # Apply transformation
    result_img = warp_affine(
        tensor_img,
        M=transform_matrix[:, :2],
        dsize=tensor_img.shape[2:],
        padding_mode=edge_handling,
        fill_value=background_value,
    )

    return result_img


def match_color_distribution(source_img, target_img):
    """Match the color distribution of source to target using LAB color space"""
    tensor_converter = ToTensor()
    image_converter = ToPILImage()

    # Convert to tensors
    source_tensor = tensor_converter(source_img).unsqueeze(0)
    target_tensor = tensor_converter(target_img).unsqueeze(0)

    # Convert to LAB color space
    source_lab = rgb_to_lab(source_tensor)
    target_lab = rgb_to_lab(target_tensor)

    # Match histograms in LAB space
    matched_lab = match_histograms(
        np.array(source_lab[0].permute(1, 2, 0)),
        np.array(target_lab[0].permute(1, 2, 0)),
        channel_axis=-1,
    )

    # Convert back to tensor and RGB
    matched_tensor = tensor_converter(matched_lab).unsqueeze(0)
    matched_rgb = lab_to_rgb(matched_tensor)
    result_img = image_converter(matched_rgb[0])

    return result_img


def extract_keyframes(prompt_text, parser=None):
    """Extract keyframe information from text with format: frame_number: prompt"""
    keyframe_list = []
    pattern = r"([0-9]+):[\s]?(.)[\S\s]"

    # Find all keyframe definitions
    keyframe_matches = re.findall(pattern, prompt_text)

    # Process each keyframe
    for frame_num, frame_prompt in keyframe_matches:
        keyframe_list.append([int(frame_num), frame_prompt])

    return keyframe_list


def detect_audio_beats(audio_path, frame_rate, audio_type):
    """Detect onset frames in audio for synchronization"""
    # Load audio file
    audio_data, sample_rate = librosa.load(audio_path)
    # Separate harmonic and percussive components if needed
    harmonic_component, percussive_component = librosa.effects.hpss(audio_data, margin=1.0)
    if audio_type == "percussive":
        audio_data = percussive_component
    elif audio_type == "harmonic":
        audio_data = harmonic_component

    # Calculate maximum possible frame
    max_frame = int((len(audio_data) / sample_rate) * frame_rate)

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(
        y=audio_data, sr=sample_rate, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1
    )
    onset_times = librosa.frames_to_time(onset_frames)

    # Convert to video frames
    frame_numbers = [int(time * frame_rate) for time in onset_times]
    frame_numbers = [0] + frame_numbers
    frame_numbers.append(max_frame)

    return {"frames": frame_numbers}


def get_beat_frames(audio_file, frames_per_second, component_type):
    """Extract audio beat frames for synchronization"""
    beat_info = detect_audio_beats(audio_file, frames_per_second, component_type)
    return beat_info["frames"]


def select_reduction_function(reduction_name):
    """Get the appropriate reduction function for mel spectrograms"""
    functions = {
        "max": np.amax,
        "median": np.median,
        "mean": np.mean
    }
    return functions.get(reduction_name)


def extract_video_info(video_file):
    """Get basic information about a video file"""
    frame_data, audio_data, metadata = read_media_file(video_file)
    total_frames = len(frame_data)
    return total_frames, metadata["video_fps"]


def spherical_interpolation(t, vector_0, vector_1, DOT_THRESHOLD=0.9995):
    """Perform spherical interpolation (slerp) between two vectors"""
    # Handle torch tensors
    using_torch = not isinstance(vector_0, np.ndarray)
    if using_torch:
        device = vector_0.device
        dtype = vector_0.dtype
    vector_0 = vector_0.cpu().numpy().astype(np.float32)
    vector_1 = vector_1.cpu().numpy().astype(np.float32)

    # Calculate vector norms
    norm_0 = np.linalg.norm(vector_0)
    norm_1 = np.linalg.norm(vector_1)

    # Calculate dot product
    normalized_dot = np.sum(vector_0 * vector_1 / (norm_0 * norm_1))

    # Choose interpolation method based on angle
    if np.abs(normalized_dot) > DOT_THRESHOLD:
        # Vectors are nearly parallel - use linear interpolation
        result = (1 - t) * vector_0 + t * vector_1
    else:
        # Use spherical interpolation
        angle_0 = np.arccos(normalized_dot)
        sin_angle_0 = np.sin(angle_0)
        angle_t = angle_0 * t
        sin_angle_t = np.sin(angle_t)
        
        s0 = np.sin(angle_0 - angle_t) / sin_angle_0
        s1 = sin_angle_t / sin_angle_0
        
        result = s0 * vector_0 + s1 * vector_1

    # Convert back to torch tensor if needed
    if using_torch:
        result = torch.from_numpy(result).to(device).to(dtype)

    return result


def create_animated_gif(image_files, output_path="./result.gif", playback_fps=24,
    quality_level=95, repeat_count=1):
    """Create an animated GIF from image files"""
    # Load all images
    frames = [Image.open(f) for f in sorted(image_files)]
    # Resize if quality reduction is needed
    if quality_level < 95:
        frames = [img.resize((128, 128), Image.LANCZOS) for img in frames]

    # Calculate frame duration
    frame_duration = len(frames) // playback_fps

    # Save as GIF
    frames[0].save(
        fp=output_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=frame_duration,
        loop=repeat_count,
        quality=quality_level,
    )


def read_media_file(filepath):
    """Read video frames and associated metadata"""
    frames, audio, metadata = read_video(
        filename=filepath, pts_unit="sec", output_format="TCHW"
    )
    return frames, audio, metadata


def create_video(image_files, output_path="./result.mp4", playback_fps=24,
    quality_level=95, audio_file=None):
    """Create a video from image files with optional audio"""
    # Load and sort images
    frames = [Image.open(f) for f in sorted(image_files, key=lambda x: x.split("/")[-1])]
    # Resize if quality reduction is needed
    if quality_level < 95:
        frames = [img.resize((128, 128), Image.LANCZOS) for img in frames]

    # Convert to tensors
    tensor_converter = ToTensor()
    frame_tensors = [tensor_converter(img) for img in frames]
    frame_tensors = [tensor.unsqueeze(0) for tensor in frame_tensors]

    # Combine into video tensor
    video_tensor = torch.cat(frame_tensors)
    video_tensor = video_tensor * 255.0
    video_tensor = video_tensor.permute(0, 2, 3, 1)
    video_tensor = video_tensor.to(torch.uint8)

    # Add audio if provided
    if audio_file is not None:
        # Load audio with correct duration
        audio_duration = len(video_tensor) / playback_fps
        audio_data, sample_rate = librosa.load(
            audio_file, sr=None, mono=True, duration=audio_duration
        )
        audio_tensor = torch.tensor(audio_data).unsqueeze(0)

        # Write video with audio
        write_video(
            output_path,
            video_array=video_tensor,
            fps=playback_fps,
            audio_array=audio_tensor,
            audio_fps=sample_rate,
            audio_codec="aac",
            video_codec="libx264",
        )
    else:
        # Write video without audio
        write_video(
            output_path,
            video_array=video_tensor,
            fps=playback_fps,
            video_codec="libx264",
        )


def store_config(output_directory, config_data):
    """Save configuration parameters to JSON file"""
    with open(f"{output_directory}/config.json", "w") as config_file:
        json.dump(config_data, config_file)


def check_xformers_availability():
    """Check if xformers acceleration is available"""
    # Check torch version (xformers has issues with torch 2.0+)
    torch_version_2 = int(torch.__version__[0]) == 2
    # Check if xformers is installed
    try:
        import xformers
        xformers_installed = True
    except (ImportError, ModuleNotFoundError):
        xformers_installed = False

    # Return True if xformers can be used
    if (not torch_version_2) and xformers_installed:
        return True

    return False
