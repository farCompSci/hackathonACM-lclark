import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoFeatureExtractor
from loguru import logger
import sys


def check_if_model_present(output_dir="./models"):
    """
    Check if the model and processor files are already present.

    Args:
        output_dir (str): Directory where models should be stored

    Returns:
        tuple: (bool, str, str) - (is_present, model_path, processor_path)
    """
    model_path = os.path.join(output_dir, "vit_model.pt")
    processor_path = os.path.join(output_dir, "vit_processor.pt")

    if os.path.exists(model_path) and os.path.exists(processor_path):
        logger.info(f"Model already present at: {model_path}")
        logger.info(f"Feature extractor already present at: {processor_path}")
        return True, model_path, processor_path

    return False, model_path, processor_path


def download_and_save_model(output_dir="./models"):
    """
    Download a model from Hugging Face using a specific commit ID and save it to disk.

    Args:
        output_dir (str): Directory to save the model

    Returns:
        tuple: (model_path, model, feature_extractor)
    """
    # Check if model is already present
    is_present, model_path, processor_path = check_if_model_present(output_dir)
    if is_present:
        # Load existing model and feature extractor
        logger.info("Loading existing model and feature extractor...")
        model = torch.load(model_path)
        feature_extractor = torch.load(processor_path)
        return model_path, model, feature_extractor

    # Use the model with specific commit ID
    model_name = "nateraw/vit-base-patch16-224-cifar10"
    commit_id = "b55eeb4"  # Using the provided commit ID

    print(f"Downloading model: {model_name} with commit ID: {commit_id}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download the feature extractor and model with specific revision
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, revision=commit_id)
        model = AutoModelForImageClassification.from_pretrained(model_name, revision=commit_id)

        # Print model info
        print(f"Model architecture: {model.__class__.__name__}")
        print(f"Number of classes: {len(model.config.id2label)}")
        print("Class labels:", model.config.id2label)

        # Save model and feature extractor
        torch.save(model, model_path)
        torch.save(feature_extractor, processor_path)

        print(f"Model saved to: {model_path}")
        print(f"Feature extractor saved to: {processor_path}")

        return model_path, model, feature_extractor

    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)