import os
import pickle

import matplotlib.pyplot as plt
from loguru import logger
from PIL import Image


def setup_image_directory(cifar_dir='./cifar-10-batches-py'):
    """
    Create the directory structure for storing images.
    If the input directory is empty, populate it with CIFAR-10 images.

    Args:
        cifar_dir (str): Path to the CIFAR-10 dataset directory

    Returns:
        tuple: (input_dir, output_dir, saliency_output_dir) - paths to input, output, and saliency output directories
    """
    # Create base image directory
    image_dir = os.path.join(os.getcwd(), "images")
    input_dir = os.path.join(image_dir, "input")
    output_dir = os.path.join(image_dir, "output")
    saliency_output_dir = os.path.join(image_dir, "saliency_output")

    # Create directories if they don't exist
    for directory in [image_dir, input_dir, output_dir, saliency_output_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

    # Check if input directory is empty
    input_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f)) and
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not input_files:
        logger.info("Input directory is empty. Populating with CIFAR-10 images...")

        # Check if the CIFAR-10 directory exists
        if os.path.exists(cifar_dir) and os.path.isdir(cifar_dir):
            populate_with_cifar10(input_dir, cifar_dir)
        else:
            logger.warning(f"CIFAR-10 directory not found at {cifar_dir}")

    else:
        logger.info(f"Found {len(input_files)} images in input directory.")

    logger.info(f"Image input directory: {input_dir}")
    logger.info(f"Image output directory: {output_dir}")
    logger.info(f"Saliency output directory: {saliency_output_dir}")

    return input_dir, output_dir, saliency_output_dir


def unpickle(file):
    """
    Unpickle the CIFAR-10 data files.

    Args:
        file (str): Path to the data batch file

    Returns:
        dict: Dictionary containing the data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10_from_files(cifar_dir, num_samples=10):
    """
    Load CIFAR-10 dataset from the downloaded batch files.

    Citation:
    The CIFAR-10 dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
    It is a subset of the 80 million tiny images dataset collected by Alex Krizhevsky,
    Vinod Nair, and Geoffrey Hinton. The dataset can be found at:
    https://www.cs.toronto.edu/~kriz/cifar.html

    Args:
        cifar_dir (str): Directory containing the CIFAR-10 batch files
        num_samples (int): Number of samples to extract per class

    Returns:
        tuple: (images, labels, class_names) where images is a list of numpy arrays,
               labels is a list of class indices, and class_names is a list of class names
    """
    logger.info(f"Loading CIFAR-10 dataset from {cifar_dir}")

    # Load meta information to get class names
    meta_file = os.path.join(cifar_dir, 'batches.meta')
    meta_data = unpickle(meta_file)
    # Convert from bytes to strings if necessary
    class_names = [label.decode('utf-8') if isinstance(label, bytes) else label
                   for label in meta_data[b'label_names']]

    logger.info(f"CIFAR-10 classes: {class_names}")

    # Load the first data batch (you can expand to use more if needed)
    batch_file = os.path.join(cifar_dir, 'data_batch_1')
    data_batch = unpickle(batch_file)

    # Get images and labels
    images = data_batch[b'data']
    labels = data_batch[b'labels']

    # Reshape images to (32, 32, 3) format
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Create a dictionary to track samples per class
    samples_per_class = {i: 0 for i in range(10)}
    selected_images = []
    selected_labels = []

    # Select num_samples from each class
    for i, (image, label) in enumerate(zip(images, labels)):
        if samples_per_class[label] < num_samples:
            selected_images.append(image)
            selected_labels.append(label)
            samples_per_class[label] += 1

        # Break if we've got enough samples for each class
        if all(count >= num_samples for count in samples_per_class.values()):
            break

    logger.info(f"Selected {len(selected_images)} images from CIFAR-10 dataset")

    return selected_images, selected_labels, class_names


def populate_with_cifar10(input_dir, cifar_dir='./cifar-10-batches-py', num_samples=10):
    """
    Populate the input directory with sample images from CIFAR-10 dataset.

    Args:
        input_dir (str): Directory to save the CIFAR-10 images
        cifar_dir (str): Directory containing the CIFAR-10 batch files
        num_samples (int): Number of samples to save from each class
    """
    logger.info(f"Loading CIFAR-10 dataset and saving {num_samples} samples per class...")

    try:
        # Load images from the downloaded CIFAR-10 files
        images, labels, class_names = load_cifar10_from_files(cifar_dir, num_samples)

        # Save images to the input directory
        saved_count = 0

        for i, (image, label) in enumerate(zip(images, labels)):
            # Create filename with class name and index
            class_name = class_names[label]
            filename = f"cifar10_{class_name}_{saved_count % num_samples}.png"
            filepath = os.path.join(input_dir, filename)

            # Save the image using PIL
            Image.fromarray(image).save(filepath)

            saved_count += 1

            if saved_count % 10 == 0:
                logger.info(f"Saved {saved_count} images so far...")

        logger.info(f"Successfully saved {saved_count} CIFAR-10 images to {input_dir}")

    except Exception as e:
        logger.error(f"Error loading or saving CIFAR-10 images: {e}")
        logger.info("Falling back to generating simple test images...")


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

