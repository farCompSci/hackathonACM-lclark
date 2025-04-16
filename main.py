from downloading_model.get_model import download_and_save_model, check_if_model_present
from model_scanning.model_scanning import scan_model_and_processor
from loguru import logger
import sys
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def setup_model():
    """
    Set up the model by either loading from disk or downloading.

    Returns:
        tuple: (model_path, processor_path, model, processor)
    """
    # Check if model is already present before downloading
    is_present, model_path, processor_path = check_if_model_present()

    if is_present:
        logger.info("Loading existing model and feature extractor...")
    else:
        logger.info("Downloading model and feature extractor...")

    model_path, model, processor = download_and_save_model()

    return model_path, processor_path, model, processor


def verify_model_security(model_path, processor_path):
    """
    Verify the security of the model and processor files.

    Args:
        model_path (str): Path to the model file
        processor_path (str): Path to the processor file

    Returns:
        bool: True if security verification passed, False otherwise
    """
    logger.info("Scanning model and processor for security vulnerabilities...")
    overall_safe, scan_results = scan_model_and_processor(model_path, processor_path)

    if not overall_safe:
        logger.warning("Security issues detected in model files!")
        user_input = input("Security vulnerabilities were found. Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            logger.info("Exiting due to security concerns.")
            return False
        logger.warning("Continuing despite security concerns...")
    else:
        logger.success("Security scan completed successfully. No vulnerabilities found.")

    return True


def setup_image_directory():
    """
    Create the directory structure for storing images.

    Returns:
        tuple: (input_dir, output_dir) - paths to input and output image directories
    """
    # Create base image directory
    image_dir = os.path.join(os.getcwd(), "images")
    input_dir = os.path.join(image_dir, "input")
    output_dir = os.path.join(image_dir, "output")

    # Create directories if they don't exist
    for directory in [image_dir, input_dir, output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    logger.info(f"Image input directory: {input_dir}")
    logger.info(f"Image output directory: {output_dir}")

    return input_dir, output_dir


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


def display_image_with_prediction(image, predicted_class, confidence, output_path=None):
    """
    Display the image with its prediction and optionally save it.

    Args:
        image: The image as a numpy array
        predicted_class (str): The predicted class name
        confidence (float): The prediction confidence
        output_path (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2%}")
    plt.axis('off')

    if output_path:
        plt.savefig(output_path)
        logger.info(f"Saved prediction visualization to {output_path}")

    plt.show()


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
    input_dir, output_dir = setup_image_directory()

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


if __name__ == '__main__':
    run()