"""
File: pipeline_foundation.py
"""
import inspect
import torch
from PIL import Image
import torchvision.transforms as Transforms
from compel import Compel, ReturnedEmbeddingsType

pil_converter = Transforms.ToPILImage("RGB")
tensor_converter = Transforms.ToTensor()

class PipelineFoundation:
    def __init__(self, diffusion_pipeline, compute_device, batch_size=1):
        self.diffusion_pipeline = diffusion_pipeline
        self.compute_device = compute_device
        self.batch_size = batch_size
        
        # Check if using SDXL model 
        self.xl_architecture = hasattr(self.diffusion_pipeline, "text_encoder_2")
        
        # Initialize conditioning processor
        if self.xl_architecture:
            self.conditioning_processor = Compel(
                tokenizer=[diffusion_pipeline.tokenizer, diffusion_pipeline.tokenizer_2],
                text_encoder=[diffusion_pipeline.text_encoder, diffusion_pipeline.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )
        else:
            self.conditioning_processor = Compel(
                tokenizer=diffusion_pipeline.tokenizer, 
                text_encoder=diffusion_pipeline.text_encoder
            )

    def prepare_image(self, input_image, target_dimensions=(512, 512)):
        """Convert image to proper format for processing"""
        img = pil_converter(input_image)
        img = img.resize(target_dimensions, resample=Image.LANCZOS)
        img = tensor_converter(img)
        return 2.0 * img - 1.0

    @staticmethod
    def convert_array_to_images(array_data):
        """
        Transform numpy array into PIL images
        """
        if array_data.ndim == 3:
            array_data = array_data[None, ...]
        array_data = (array_data * 255).round().astype("uint8")
        
        if array_data.shape[-1] == 1:
            # Handle grayscale images
            result_images = [
                Image.fromarray(single_array.squeeze(), mode="L") for single_array in array_data
            ]
        else:
            result_images = [Image.fromarray(single_array) for single_array in array_data]

        return result_images

    @staticmethod
    def convert_array_to_tensor(array_data):
        """
        Transform numpy array into pytorch tensor
        """
        if array_data.ndim == 3:
            array_data = array_data[..., None]

        tensor_data = torch.from_numpy(array_data.transpose(0, 3, 1, 2))
        return tensor_data

    @staticmethod
    def convert_tensor_to_array(tensor_data):
        """
        Transform pytorch tensor into numpy array
        """
        array_data = tensor_data.cpu().permute(0, 2, 3, 1).float().numpy()
        return array_data

    def finalize_image(self, image_data, output_format: str = "pil"):
        """Process output to desired format"""
        if isinstance(image_data, torch.Tensor) and output_format == "pt":
            return image_data

        if isinstance(image_data, torch.Tensor):
            image_data = self.convert_tensor_to_array(image_data)

        if output_format == "np":
            return image_data
        elif output_format == "pil":
            return self.convert_array_to_images(image_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}.")

    @torch.no_grad()
    def latent_to_image(self, encoded_data):
        """Convert latent representation to image"""
        return self.diffusion_pipeline.decode_latents(encoded_data)

    @torch.no_grad()
    def image_to_latent(self, input_image, rand_generator=None):
        """Convert image to latent representation"""
        latent_distribution = self.diffusion_pipeline.vae.encode(input_image.to(self.compute_device)).latent_dist
        latent_sample = 0.18215 * latent_distribution.sample(generator=rand_generator)
        return latent_sample

    @torch.no_grad()
    def text_to_conditioning(self, prompt_text):
        """Convert text prompt to embedding conditioning"""
        if self.xl_architecture:
            text_embedding, pooled_embedding = self.conditioning_processor(prompt_text)
            result = {"text_embeddings": text_embedding, "pooled_embeddings": pooled_embedding}
        else:
            text_embedding = self.conditioning_processor(prompt_text)
            result = {"text_embeddings": text_embedding}

        return result

    @torch.no_grad()
    def process_noise_prediction(self, latent_input, text_embedding, iteration, timestep, guidance_strength):
        """Process noise prediction using guidance"""
        # Check for eta parameter acceptance in scheduler
        supports_eta = "eta" in set(
            inspect.signature(self.diffusion_pipeline.scheduler.step).parameters.keys()
        )
        step_kwargs = {"eta": 0.0} if supports_eta else {}

        # Prepare latent input with proper batch shape
        model_input = torch.cat(
            list(
                map(
                    lambda latent, text_embed: torch.cat(
                        [latent] * text_embed.shape[0]
                    ),
                    latent_input.chunk(self.batch_size),
                    text_embedding.chunk(self.batch_size),
                )
            )
        )

        # Create unconditional embedding
        max_length = text_embedding.shape[1]
        empty_prompt = self.diffusion_pipeline.tokenizer(
            [""] * model_input.shape[0],
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embedding = self.diffusion_pipeline.text_encoder(
            empty_prompt.input_ids.to(self.compute_device)
        )[0]
        
        # Combine conditional and unconditional for classifier-free guidance
        combined_embedding = torch.cat([uncond_embedding, text_embedding])
        combined_input = torch.cat([model_input] * 2)

        # Get model prediction
        prediction = self.diffusion_pipeline.unet(
            combined_input, timestep, encoder_hidden_states=combined_embedding
        )["sample"]

        # Split prediction and apply guidance
        uncond_pred, cond_pred = prediction.chunk(2)
        
        # Average predictions across batch elements
        uncond_pred_avg = torch.cat(
            list(
                map(
                    lambda x: x.mean(dim=0, keepdim=True),
                    uncond_pred.chunk(self.batch_size),
                )
            )
        )
        cond_pred_avg = torch.cat(
            list(
                map(
                    lambda x: x.mean(dim=0, keepdim=True),
                    cond_pred.chunk(self.batch_size),
                )
            )
        )

        # Apply classifier-free guidance
        guided_prediction = uncond_pred_avg + guidance_strength * (
            cond_pred_avg - uncond_pred_avg
        )
        
        # Get next latent state
        next_latent = self.diffusion_pipeline.scheduler.step(guided_prediction, timestep, latent_input, **step_kwargs)[
            "prev_sample"
        ]

        return next_latent

    @torch.no_grad()
    def generate_with_latent(
        self,
        conditioning_embedding,
        initial_latent,
        diffusion_steps=50,
        guidance_strength=7.5,
        step_offset=1,
        noise_param=0.0,
    ):
        """Run the diffusion process on a latent input"""
        # Configure scheduler
        self.diffusion_pipeline.scheduler.set_timesteps(diffusion_steps)
        self.diffusion_pipeline.scheduler.config.steps_offset = 1

        # Scale latent by noise level
        working_latent = initial_latent * self.diffusion_pipeline.scheduler.init_noise_sigma

        # Check for eta parameter in scheduler
        supports_eta = "eta" in set(
            inspect.signature(self.diffusion_pipeline.scheduler.step).parameters.keys()
        )
        step_kwargs = {"eta": noise_param} if supports_eta else {}

        # Run diffusion process
        for step_idx, current_time in enumerate(self.diffusion_pipeline.scheduler.timesteps):
            # Scale input according to scheduler
            scaled_latent = self.diffusion_pipeline.scheduler.scale_model_input(working_latent, current_time)
            # Process through prediction and guidance
            working_latent = self.process_noise_prediction(
                scaled_latent, conditioning_embedding, step_idx, current_time, guidance_strength
            )

        return working_latent

    def balance_embeddings(self, embedding_a, embedding_b):
        """Ensure embeddings have compatible dimensions by padding as needed"""
        if embedding_a.shape == embedding_b.shape:
            return embedding_a, embedding_b

        # Determine which embedding needs padding
        smaller_embed = min(
            [embedding_a, embedding_b],
            key=lambda key: key.shape[0],
        )
        larger_embed = max(
            [embedding_a, embedding_b],
            key=lambda key: key.shape[0],
        )
        # Calculate difference in size
        size_difference = larger_embed.shape[0] - smaller_embed.shape[0]

        # Create padding using empty string embeddings
        empty_embedding = torch.cat([self.text_to_conditioning(self.diffusion_pipeline, "")] * size_difference)
        
        # Apply padding to appropriate embedding
        if embedding_a.shape[0] < embedding_b.shape[0]:
            embedding_a = torch.cat([embedding_a, empty_embedding])
        else:
            embedding_b = torch.cat([embedding_b, empty_embedding])

        return embedding_a, embedding_b

    def run_safety_check(self, image_array):
        """Verify images with safety checker if available"""
        pil_images = self.convert_array_to_images(image_array)

        # Extract features for safety checking
        features = self.diffusion_pipeline.feature_extractor(
            pil_images, return_tensors="pt"
        ).to(self.compute_device)

        # Run safety checker
        filtered_images, nsfw_flags = self.diffusion_pipeline.safety_checker(
            images=image_array, clip_input=features.pixel_values
        )

        return filtered_images, nsfw_flags

