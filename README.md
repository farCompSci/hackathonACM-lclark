# Needle in a Haystack: Finding Critical Pixels in Images

## Project Overview

This project explores the concept of finding the "needle in a haystack" within image classification by identifying the most important pixels that influence how a computer vision model classifies images. Just as finding a needle in a haystack is challenging due to the overwhelming amount of hay compared to the tiny needle, finding the critical pixels that determine classification is difficult due to the large number of pixels in an image. 

### Why Needle in a Haystack?

The idea of needle in a haystack was the theme of a week-long hackathon project that started out on Thursday, April 10th, 2025 at Lewis & Clark College, as organized by the ACM Club. 

## What Does This Project Do?

This application:

1. **Loads a pre-trained Vision Transformer (ViT) model** that can classify images into various categories
2. **Analyzes images** using saliency maps to understand which parts of the image are most important for classification
3. **Identifies critical pixels** - the individual points in the image that have the most influence on how the model "sees" the image
4. **Visualizes results** with easy-to-understand heatmaps highlighting important regions

## Technical Concepts Explained

### CIFAR-10 Dataset

The CIFAR-10 dataset is a collection of 60,000 small color images (32x32 pixels) divided into 10 different classes:
- Airplanes
- Automobiles
- Birds
- Cats
- Deer
- Dogs
- Frogs
- Horses
- Ships
- Trucks

This dataset was collected by researchers Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton, and is widely used for training and testing computer vision models. Think of it as a standardized test for image classification algorithms.

### Vision Transformer (ViT)

While traditional convolutional neural networks (CNNs) analyze images by looking at small patches and gradually building up understanding, Vision Transformers work differently:

1. They divide an image into a grid of small patches (like cutting a photo into squares)
2. They process these patches simultaneously, allowing them to understand relationships between distant parts of the image
3. They use an "attention mechanism" to focus on important parts of the image

This approach is similar to how humans might quickly scan an entire image and focus on the most interesting or informative parts.

### Saliency Maps

A saliency map highlights the regions of an image that the model is focusing on when making its decision. Similar to a heat map:
- Bright/hot areas indicate regions the model considers important
- Dark/cool areas are regions the model largely ignores

In our project, we specifically use the attention mechanism of the Vision Transformer to create these maps, showing which parts of the image are receiving the most "attention" from the model.

### Critical Pixels

While saliency maps show important regions, our "needle in the haystack" approach goes further to identify individual pixels that have the most influence on classification decisions. These critical pixels represent the "needles" - the tiny but crucial elements that determine how the model interprets the entire image.

## How to Use This Project

1. **Setup Environment**:
`pip install -r requirements.txt`

2. **Run the Application**:
`python3 main.py`

3. **Input image index of the image**:
You will be prompted for this one. Just refer to images/input, where the first image will have an index of `0`. 

5. **Interpret Results**:
- The application will create three kinds of visualizations in separate folders:
  - `images/output`: Basic predictions
  - `images/saliency_output`: Visualizations showing:
    - The original image
    - The saliency map
    - The original image with critical pixels highlighted

## Project Structure
|─ downloading_models/                &emsp; &emsp; &emsp; &emsp; # Package for model setup <br/>
|─ handle_images/                     &emsp; &emsp; &emsp; &emsp; # Package for image processing<br/>
|─ model_scanning/                    &emsp; &emsp; &emsp; &emsp; # Package for model analysis and saliency maps<br/>
|─ images/                            &emsp; &emsp; &emsp; &emsp; &emsp; # Storage for images <br/>
&emsp;    └─ input/                         &emsp; &emsp; &emsp; &emsp; &emsp; # Input images (CIFAR-10 or user-provided)<br/>
&emsp;    └─ output/                        &emsp; &emsp; &emsp; &emsp; &emsp; # Basic prediction visualizations<br/>
&emsp;    └─ saliency_output/               &emsp; &emsp; &emsp; &emsp; &emsp; # Saliency map visualizations<br/>
|─ models/                            &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # Storage for model files<br/>
|─ main.py                            &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; # Main entry point<br/>
|─ requirements.txt                   &emsp; &emsp; &emsp; &emsp; # Dependencies<br/>

## Future Enhancements

Future versions of this project could include:
- One-pixel attack implementation (changing a single pixel to alter classification)
- Interactive visualization allowing users to select different images
- Comparison of different visualization techniques
- Analysis of model robustness based on critical pixel modification

## Acknowledgments

This project uses the CIFAR-10 dataset collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. The version used was fetched here: https://www.cs.toronto.edu/~kriz/cifar.html . <br/>
This project also uses ProtectAI's model scanner to ensure model security before usage. This can be found here: https://github.com/protectai/modelscan/tree/main . <br/>
The model used was fetched from huggingface with `model_id`: `nateraw/vit-base-patch16-224-cifar10`. <br/>
All other dependecies used can be found in the `requirements.txt`, which include the packages and libraries I do not claim to have made.

