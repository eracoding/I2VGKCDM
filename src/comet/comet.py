"""
---------------------------------
File: comet.py
Author: Ulugbek Shernazarov
Email: u.shernaz4rov@gmail.com
Copyright (c) 2025 Ulugbek Shernazarov. All rights reserved | GitHub: eracoding
Description: Comet ML integration for tracking and monitoring image generation experiments. Provides a wrapper class for managing experiment lifecycles, logging parameters, metrics, assets, and models, with specialized support for tracking images, videos, and other visualization outputs.
---------------------------------
"""
import os
import tempfile
import numpy as np
from PIL import Image
import comet_ml
from comet_ml import Experiment
import torch
import io

class CometTracker:
    def __init__(self, project_name="I2VGKCDM", workspace=None, api_key=None):
        """
        Initialize Comet ML tracking.
        
        Args:
            project_name: Name of the Comet ML project
            workspace: Comet ML workspace name
            api_key: Comet ML API key
        """
        self.project_name = project_name
        self.workspace = workspace
        
        # Use the provided API key or get from environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("COMET_API_KEY")
        
        self.experiment = None
    
    def start_experiment(self, experiment_name=None, tags=None):
        """Start a Comet ML experiment."""
        self.experiment = Experiment(
            api_key=self.api_key,
            project_name=self.project_name,
            workspace=self.workspace
        )
        
        if experiment_name:
            self.experiment.set_name(experiment_name)
        
        if tags:
            self.experiment.add_tags(tags)
        
        return self.experiment
    
    def end_experiment(self):
        """End the current Comet ML experiment."""
        if self.experiment:
            self.experiment.end()
            self.experiment = None
    
    def log_parameters(self, params):
        """Log parameters to the current experiment."""
        if self.experiment:
            self.experiment.log_parameters(params)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to the current experiment."""
        if self.experiment:
            for name, value in metrics.items():
                self.experiment.log_metric(name, value, step=step)
    
    def log_image(self, image, name, step=None):
        """
        Log a PIL Image or numpy array to the current experiment.
        
        Args:
            image: PIL Image or numpy array
            name: Name of the image
            step: Current step (optional)
        """
        if self.experiment:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            self.experiment.log_image(image, name=name, step=step)
    
    def log_images(self, images, prefix="frame", step=None):
        """
        Log multiple images to the current experiment.
        
        Args:
            images: List of PIL Images
            prefix: Prefix for image names
            step: Current step (optional)
        """
        if self.experiment:
            for i, img in enumerate(images):
                name = f"{prefix}_{i:04d}"
                self.log_image(img, name=name, step=step)
    
    def log_video(self, video_path, name=None):
        """Log a video to the current experiment."""
        if self.experiment:
            self.experiment.log_video(video_path, name=name)
    
    def log_gif(self, gif_path, name=None):
        """Log a GIF to the current experiment."""
        if self.experiment:
            self.experiment.log_asset(gif_path, name=name if name else "animation.gif")
    
    def log_model(self, model, name=None):
        """Log a PyTorch model to the current experiment."""
        if self.experiment:
            if isinstance(model, torch.nn.Module):
                with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp:
                    torch.save(model.state_dict(), temp.name)
                    self.experiment.log_model(name if name else "model", temp.name)
                    os.unlink(temp.name)
            else:
                raise ValueError("Model must be a PyTorch nn.Module")
    
    def log_table(self, table, name):
        """Log a pandas DataFrame or list-based table to the current experiment."""
        if self.experiment:
            self.experiment.log_table(name, table)
    
    def log_confusion_matrix(self, matrix, title=None):
        """Log a confusion matrix to the current experiment."""
        if self.experiment:
            self.experiment.log_confusion_matrix(matrix=matrix, title=title)
    
    def log_code(self, file_path=None):
        """Log code files to the current experiment."""
        if self.experiment:
            if file_path:
                self.experiment.log_code(file_path)
            else:
                self.experiment.log_code()

def start_experiment():
    try:
        api = comet_ml.API()

        workspace = comet_ml.config.get_config()["comet.workspace"]
        if workspace is None:
            workspace = api.get_default_workspace()

        project_name = comet_ml.config.get_config()["comet.project_name"]

        experiment = comet_ml.APIExperiment(
            workspace=workspace, project_name=project_name
        )
        experiment.log_other("Created from", "stable-diffusion")
        return experiment

    except Exception as e:
        return None
