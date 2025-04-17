import os
import sys

import torch
from loguru import logger
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModelForImageClassification


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

    # First, try to set up the safe globals for loading
    try:
        # Import the ViT class properly
        from transformers.models.vit.modeling_vit import ViTForImageClassification

        # Add it to the safe globals list
        torch.serialization.add_safe_globals([ViTForImageClassification])
        logger.info("Added ViTForImageClassification to safe globals")
    except Exception as e:
        logger.warning(f"Could not add ViTForImageClassification to safe globals: {e}")

    if is_present:
        # Load existing model and feature extractor
        logger.info("Loading existing model and feature extractor...")
        try:
            # First try with safe globals setup
            try:
                model = torch.load(model_path)
                processor = torch.load(processor_path)
            except Exception as e1:
                logger.warning(f"Could not load with default settings: {e1}")
                # Fallback to weights_only=False (less secure but needed for compatibility)
                logger.info("Trying with weights_only=False...")
                model = torch.load(model_path, weights_only=False)
                processor = torch.load(processor_path, weights_only=False)

            return model_path, model, processor

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Re-downloading model...")
            # Continue with download if loading fails

    # If we get here, model needs to be downloaded
    # Use the model with specific commit ID
    model_name = "nateraw/vit-base-patch16-224-cifar10"
    commit_id = "b55eeb4"  # Using the provided commit ID

    logger.info(f"Downloading model: {model_name} with commit ID: {commit_id}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Download the feature extractor and model with specific revision
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, revision=commit_id)
        model = AutoModelForImageClassification.from_pretrained(model_name, revision=commit_id)

        # Print model info
        logger.info(f"Model architecture: {model.__class__.__name__}")
        logger.info(f"Number of classes: {len(model.config.id2label)}")
        logger.info(f"Class labels: {model.config.id2label}")

        # Don't save the model to disk since loading it back is problematic
        # Instead, just return the model directly
        logger.info("Using model directly from memory (skipping save/load cycle to avoid unpickling issues)")

        return model_path, model, feature_extractor

    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        sys.exit(1)


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
