"""
---------------------------------
File: mlflow_utils.py
Author: Ulugbek Shernazarov
Email: u.shernaz4rov@gmail.com
Copyright (c) 2025 Ulugbek Shernazarov. All rights reserved | GitHub: eracoding
Description: MLflow integration utilities for experiment tracking. Provides a wrapper class for managing experiment runs, logging parameters, metrics, artifacts, and models with support for tracking images, videos, and other generation outputs.
---------------------------------
"""
import os
import mlflow
import tempfile
from PIL import Image
import numpy as np
import io

class MLflowTracker:
    def __init__(self, experiment_name="image_generation", tracking_uri=None):
        """
        Initialize MLflow tracking.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Check for environment variable
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(mlflow_uri)
        
        # Set or create the experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        self.run = None
    
    def start_run(self, run_name=None, tags=None):
        """Start an MLflow run."""
        self.run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name, tags=tags)
        return self.run
    
    def end_run(self):
        """End the current MLflow run."""
        if self.run:
            mlflow.end_run()
            self.run = None
    
    def log_params(self, params):
        """Log parameters to the current run."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to the current run."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path, artifact_path=None):
        """Log an artifact to the current run."""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_dict(self, dictionary, artifact_file):
        """Log a dictionary as an artifact."""
        mlflow.log_dict(dictionary, artifact_file)
    
    def log_figure(self, figure, artifact_file):
        """Log a matplotlib figure as an artifact."""
        mlflow.log_figure(figure, artifact_file)
    
    def log_image(self, image, artifact_file):
        """
        Log a PIL Image or numpy array as an artifact.
        
        Args:
            image: PIL Image or numpy array
            artifact_file: Name of the artifact file
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp:
            image.save(temp.name)
            mlflow.log_artifact(temp.name, artifact_file)
            os.unlink(temp.name)
    
    def log_images(self, images, artifact_dir="images"):
        """
        Log multiple images as artifacts.
        
        Args:
            images: List of PIL Images
            artifact_dir: Directory to store images within artifacts
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, img in enumerate(images):
                img_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                img.save(img_path)
            
            mlflow.log_artifacts(temp_dir, artifact_dir)
    
    def log_video(self, video_path, artifact_path=None):
        """Log a video as an artifact."""
        mlflow.log_artifact(video_path, artifact_path)
    
    def log_model(self, model, artifact_path):
        """Log a model as an artifact."""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def register_model(self, model_uri, name):
        """Register a model."""
        return mlflow.register_model(model_uri, name)
