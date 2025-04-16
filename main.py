from downloading_model.get_model import download_and_save_model, check_if_model_present
from model_scanning.model_scanning import scan_model_and_processor
from loguru import logger
import sys
import torch


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


def run():
    logger.info("Starting Needle in a Haystack Pipeline")

    # Setup model
    model_path, processor_path, model, processor = setup_model()

    # Verify model security
    if not verify_model_security(model_path, processor_path):
        sys.exit(1)

    logger.info("Model preparation and security scanning complete!")


if __name__ == '__main__':
    run()