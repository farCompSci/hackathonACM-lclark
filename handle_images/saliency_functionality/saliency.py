import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image


class SaliencyMapGenerator:
    """
    Class to generate saliency maps for Vision Transformer models.
    Uses attention mechanism to identify important pixels.
    """

    def __init__(self, model, processor):
        """
        Initialize the saliency map generator.

        Args:
            model: The Vision Transformer model
            processor: The image processor
        """
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device

    def generate_attention_based_saliency(self, image_path, target_class=None):
        """
        Generate a saliency map based on the attention weights from the ViT model.

        Args:
            image_path (str): Path to the input image
            target_class (int, optional): Target class for saliency map. If None, uses predicted class.

        Returns:
            tuple: (saliency_map, predicted_class, prediction_confidence, attention_weights)
        """
        logger.info(f"Generating attention-based saliency map for {image_path}")

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get the prediction and attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            logits = outputs.logits
            attentions = outputs.attentions  # attention weights

            # Get the predicted class
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            pred_class_idx = probs.argmax().item() if target_class is None else target_class
            pred_class = self.model.config.id2label[pred_class_idx]
            confidence = probs[pred_class_idx].item()

            logger.info(f"Predicted class: {pred_class} with confidence: {confidence:.2%}")

        # Process attention weights to create saliency map
        # Get the last layer's attention (typically most informative)
        last_layer_attention = attentions[-1].mean(dim=1)  # Average over attention heads

        # Extract attention for the [CLS] token to all other tokens
        # This shows which image patches the model attends to for classification
        cls_attention = last_layer_attention[0, 0,
                        1:]  # [0, 0, 1:] = first sample, first token ([CLS]), all patch tokens

        # Reshape attention to match the image patches
        # ViT typically uses 16x16 patches, so we need to reshape based on that
        num_patches = int(np.sqrt(cls_attention.shape[0]))
        attention_map = cls_attention.reshape(num_patches, num_patches).cpu().numpy()

        # Upscale the attention map to original image size
        saliency_map = self._upscale_attention(attention_map, original_size)

        return saliency_map, pred_class, confidence, attention_map

    def _upscale_attention(self, attention_map, target_size):
        """
        Upscale the attention map to match the original image dimensions.

        Args:
            attention_map (numpy.ndarray): The attention map
            target_size (tuple): Target size (width, height)

        Returns:
            numpy.ndarray: Upscaled attention map
        """
        # Convert to PIL image for resizing
        attention_pil = Image.fromarray(
            (attention_map * 255 / np.max(attention_map)).astype(np.uint8)
        )

        # Resize to match original image
        resized_attention = attention_pil.resize(target_size, Image.BICUBIC)

        # Convert back to numpy array
        return np.array(resized_attention) / 255.0

    def find_critical_pixels(self, saliency_map, top_n=5):
        """
        Find the top N most important pixels based on the saliency map.

        Args:
            saliency_map (numpy.ndarray): The saliency map
            top_n (int): Number of top pixels to return

        Returns:
            list: List of (x, y, importance) tuples for top pixels
        """
        # Flatten the map to find top values
        flat_indices = np.argsort(saliency_map.flatten())[-top_n:]

        # Convert flat indices to 2D coordinates
        height, width = saliency_map.shape
        critical_pixels = []

        for idx in flat_indices:
            y, x = divmod(idx, width)
            importance = saliency_map[y, x]
            critical_pixels.append((x, y, importance))

        # Sort by importance (highest first)
        critical_pixels.sort(key=lambda p: p[2], reverse=True)

        return critical_pixels

    def visualize_saliency(self, image_path, saliency_map, critical_pixels=None,
                           pred_class=None, confidence=None, output_path=None):
        """
        Visualize the original image, saliency map, and critical pixels.

        Args:
            image_path (str): Path to the original image
            saliency_map (numpy.ndarray): The saliency map
            critical_pixels (list, optional): List of critical pixels
            pred_class (str, optional): Predicted class name
            confidence (float, optional): Prediction confidence
            output_path (str, optional): Path to save the visualization

        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Load the original image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)

        # Create a figure with multiple subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot original image
        axes[0].imshow(image_array)
        title = "Original Image"
        if pred_class and confidence:
            title += f"\nPredicted: {pred_class}\nConfidence: {confidence:.2%}"
        axes[0].set_title(title)
        axes[0].axis('off')

        # Plot saliency map
        axes[1].imshow(saliency_map, cmap='hot')
        axes[1].set_title("Saliency Map")
        axes[1].axis('off')

        # Plot original image with critical pixels highlighted
        axes[2].imshow(image_array)
        if critical_pixels:
            for i, (x, y, importance) in enumerate(critical_pixels):
                # Plot marker at critical pixel
                axes[2].plot(x, y, 'o', markersize=10,
                             markeredgecolor='white', markerfacecolor='none')

                # Add rank text : uncomment if you want rank of importance
                # axes[2].text(x + 5, y + 5, f"#{i + 1}", color='white',
                #              fontsize=12, fontweight='bold')

        axes[2].set_title("Critical Pixels")
        axes[2].axis('off')

        plt.tight_layout()

        # Save if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved saliency visualization to {output_path}")

        return fig

    def run_saliency_analysis(self, image_path, saliency_output_dir, top_n=5):
        """
        Run the complete saliency analysis pipeline for an image.

        Args:
            image_path (str): Path to the input image
            saliency_output_dir (str): Directory to save saliency outputs
            top_n (int): Number of top critical pixels to identify

        Returns:
            tuple: (saliency_map, critical_pixels, predicted_class, confidence)
        """
        # Generate the saliency map
        saliency_map, pred_class, confidence, _ = self.generate_attention_based_saliency(image_path)

        # Find critical pixels
        critical_pixels = self.find_critical_pixels(saliency_map, top_n=top_n)

        # Log critical pixels
        logger.info(f"Top {len(critical_pixels)} critical pixels:")
        for i, (x, y, importance) in enumerate(critical_pixels):
            logger.info(f"  #{i + 1}: Position: ({x}, {y}), Importance: {importance:.4f}")

        # Create output file path
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(saliency_output_dir, f"saliency_{base_filename}.png")

        # Visualize and save
        self.visualize_saliency(
            image_path, saliency_map, critical_pixels,
            pred_class, confidence, output_path
        )

        return saliency_map, critical_pixels, pred_class, confidence