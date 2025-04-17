import os
import sys

import numpy as np
import torch
from loguru import logger
from PIL import Image

from handle_images.image_helpers.image_helpers import (
    display_image_with_prediction,
    setup_image_directory,
)
from handle_images.saliency_functionality.saliency_helpers import run_saliency_analysis
from handle_model_setup.setup_model import setup_model
from model_scanning.model_scanning import verify_model_security


def run_inference(model, processor, image_path):
    """
    Run inference on an image using the loaded model.

    Args:
        model: The loaded vision model
        processor: The feature extractor or processor
        image_path (str): Path to the image file

    Returns:
        tuple: (image_tensor, original_image, predicted_class, confidence, class_idx)
    """
    # Check if the image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found at {image_path}")
        return None

    logger.info(f"Running inference on image: {image_path}")

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)

    # Process image
    inputs = processor(images=image, return_tensors="pt")

    # Move to the appropriate device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

        # Get the predicted class
        class_idx = probabilities.argmax().item()
        predicted_class = model.config.id2label[class_idx]
        confidence = probabilities[class_idx].item()

    logger.info(f"Prediction: {predicted_class}")
    logger.info(f"Confidence: {confidence:.2%}")

    # Display top 3 predictions
    top_3_idx = torch.topk(probabilities, 3).indices.tolist()
    for idx in top_3_idx:
        class_name = model.config.id2label[idx]
        prob = probabilities[idx].item()
        logger.info(f"  {class_name}: {prob:.2%}")

    return inputs, original_image, predicted_class, confidence, class_idx


def run():
    """
    Run the complete Needle in a Haystack pipeline.
    """
    logger.info("Starting Needle in a Haystack Pipeline")

    # Setup model
    model_path, processor_path, model, processor = setup_model()

    # Verify model security
    if not verify_model_security(model_path, processor_path):
        sys.exit(1)

    logger.info("Model preparation and security scanning complete!")

    # Setup image directories
    input_dir, output_dir, saliency_output_dir = setup_image_directory()

    # Look for images in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        logger.warning(f"No images found in {input_dir}")
        logger.info(f"Please add images to {input_dir} directory")
        sys.exit(0)

    # Select an image (first one for now, can be made interactive)
    image_path = os.path.join(input_dir, image_files[0])
    logger.info(f"Selected image: {image_files[0]}")

    # Run inference
    inference_result = run_inference(model, processor, image_path)

    if inference_result:
        inputs, original_image, predicted_class, confidence, class_idx = inference_result

        # Display and save the prediction visualization
        output_filename = f"prediction_{os.path.splitext(image_files[0])[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        display_image_with_prediction(original_image, predicted_class, confidence, output_path)

        # Run saliency map analysis
        saliency_map, critical_pixels = run_saliency_analysis(model, processor, image_path, saliency_output_dir)

        logger.info("Pipeline completed successfully!")

if __name__ == '__main__':
    run()