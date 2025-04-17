import os

from loguru import logger

from .saliency import SaliencyMapGenerator


def run_saliency_analysis(model, processor, image_path, saliency_output_dir):
    """
    Run saliency map analysis on an image.

    Args:
        model: The vision model
        processor: The feature extractor or processor
        image_path (str): Path to the image file
        saliency_output_dir (str): Directory to save saliency visualizations

    Returns:
        tuple: (saliency_map, critical_pixels)
    """
    logger.info("Running saliency map analysis...")

    # Initialize the saliency map generator
    saliency_generator = SaliencyMapGenerator(model, processor)

    # Run the analysis
    saliency_map, critical_pixels, pred_class, confidence = (
        saliency_generator.run_saliency_analysis(image_path, saliency_output_dir)
    )

    logger.info(f"Saliency analysis complete for {os.path.basename(image_path)}")
    logger.info(f"Predicted class: {pred_class}, Confidence: {confidence:.2%}")

    return saliency_map, critical_pixels