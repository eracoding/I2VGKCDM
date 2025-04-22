# I2VGKCDM

**Image-to-Video Generation using Kinematic Conditioning and Diffusion Models**

I2VGKCDM is a powerful and flexible toolkit for generating videos or animated GIFs from text prompts using Stable Diffusion. It supports a wide range of features including smooth interpolation between frames, experiment tracking, and both CLI and web interfaces.

---

## Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/eracoding/I2VGKCDM/blob/main/I2VGKCDM.ipynb)

## Features

- Generate sequences of related images from text prompts
- Create smooth transitions between images by controlling variation
- Export as individual frames, videos, or GIFs
- Simple command-line interface
- Configurable parameters for customization

## Project Structure
```
I2VGKCDM/
├── src/                         # Core implementation files
|   ├── pipelines/               # All pipelines for generation
|   |   ├── custom_pipeline.py   # A customizable diffusion pipeline for image and video generation that supports various input sources
|   |   └── pipeline_base.py     # Foundation class for diffusion model pipelines that handles core operations such as text-to-conditioning conversions
|   ├── comet/                   # Comet ML configuration
|   ├── preprocess               # Preprocessing package
|   ├── generate.py              # Core module for generating image sequences using diffusion models
|   ├── helpers.py               # Utility functions for image processing, video handling, and animation support
|   ├── session.py               # Session management module for saving and loading generation parameters
|   └── mlflow_utils.py          # MLflow integration utilities for experiment tracking
├── main.py                      # Main application entry point for the image animation generation system via gradio
├── run_generation.py            # Command-line interface
├── webapp.py                    # Streamlit web interface
├── configs/                     # Configuration files for experiments
│   └── experiments.json         # Sample experiment configurations
├── README.md                    # Project overview
├── poetry.lock                  # Poetry lock file
├── pyproject.toml               # Poetry dependencies
├── Dockerfile                   # Docker file to run the project with docker tool
├── docker-compose.yaml          # Docker orchestration
└── requirements.txt             # Dependencies
```

---

### Demo
![](https://github.com/eracoding/I2VGKCDM/blob/main/media/demo.gif)

---

## Getting Started

### Option 1: Manual Setup (Virtual Environment)

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### Option 2: Poetry Setup (Recommended)

#### Install Poetry

If you don't have Poetry installed, run:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then restart your shell and verify installation:

```bash
poetry --version
```

#### Set up the project environment

```bash
# Navigate to the project directory
cd I2VGKCDM

# Install dependencies and create virtual environment
poetry install

# Activate the environment
poetry shell
```

---

## Optional: GPU Acceleration with PyTorch

If you have a CUDA-compatible GPU, install the appropriate PyTorch version for better performance:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Usage Instructions

### Main application

Run gradio application implementation:

```bash
poetry run python main.py
```

### Basic CLI Usage

Generate a video from a text prompt:

```bash
python run_generation.py --prompt "A futuristic cityscape at sunset" --n_frames 16 --fps 8
```

---

### Streamlit Web Interface

Launch the web app:

```bash
streamlit run webapp.py
```

A browser window will open with an interactive interface for prompt-based generation.

---

### Running Experiments

#### Run a single experiment:

```bash
python run_generation.py --config configs/experiments.json
```

#### Run a parameter sweep:

```bash
python run_generation.py --config configs/experiments.json --sweep
```

---

## Docker Support (Optional)

To run the project inside a Docker container:

```bash
docker-compose up --build
```

---

## Contributions

Contributions, issues, and suggestions are welcome! Feel free to open a PR or issue.

---

## License

MIT License. See `LICENSE` file for details.
