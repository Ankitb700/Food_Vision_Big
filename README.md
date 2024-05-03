# Food Vision with Vision Transformer and PyTorch

The Food Vision project aims to create an image classification system that identifies different types of food from a large dataset (Food-101). It leverages PyTorch, a deep learning framework, and Vision Transformer (ViT), a state-of-the-art model architecture for image classification.

## Introduction

The Food Vision project uses Vision Transformer (ViT) in PyTorch to classify images of various food items from the Food-101 dataset. This dataset contains 101,000 images across 101 different food categories. The project involves training a ViT model to accurately classify these food images and evaluate its performance.

## Technologies Used

- **Python**: The project is implemented in Python, a popular programming language for machine learning and deep learning tasks.
- **PyTorch**: PyTorch is used to build, train, and evaluate the ViT model for image classification.
- **Vision Transformer (ViT)**: ViT is a transformer-based architecture designed for image classification tasks. It applies attention mechanisms to process images in a unique way.
- **Matplotlib**: This plotting library is used to visualize model performance and results.
- **Gradio**: A web framework for creating interactive applications. It can be used to build a user interface for model inference.

## Model Architecture

The model architecture for the Food Vision project involves the following components:

- **Vision Transformer (ViT)**: The backbone of the model. It applies self-attention mechanisms to image patches to generate meaningful representations.
- **Classification Head**: The final layer that outputs the predicted food category. This head is connected to the ViT model and classifies images into 101 categories.

## Usage

To use the Food Vision project:

1. **Dataset Preparation**: Download and prepare the Food-101 dataset. This involves splitting the dataset into training and testing sets.
2. **Model Training**: Train the ViT model on the training dataset. Adjust hyperparameters such as learning rate and batch size as needed.
3. **Model Evaluation**: Evaluate the trained model on the testing dataset to measure its performance.
4. **Model Inference**: Use the trained model to classify new images of food.
5. **Gradio Deployment**: Optionally, deploy the system using Gradio to provide a user-friendly interface for interacting with the model.

## Example Code

Here is a simple example of how to train a ViT model with the Food-101 dataset in PyTorch:

```python
import torch
import torchvision

from torch import nn


def create_effnetb2_model(num_classes:int=3, 
                          seed:int=42):
    """Creates an EfficientNetB2 feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 3.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB2 feature extractor model. 
        transforms (torchvision.transforms): EffNetB2 image transforms.
    """
    # Create EffNetB2 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=1408, out_features=num_classes),
    )
    
    return model, transforms

    
## Screenshots

![App Screenshot](https://github.com/Ankitb700/Food_Vision_Big/blob/main/Images/Screenshot%20(145).png)


## Deployment

To deploy this project run

```bash
  python app.run
```


## Tech Stack

**Client:** Gradio

**Tech:** Python,matplotlib,pandas,numpy,data analysis,mahine learning,pytorch


## Documentation

```
# FoodVisionBig

FoodVisionBig is a project that allows users to upload an image and retrieve the name of the food item depicted in the image. It utilizes a pre-trained deep learning model trained on the Food-101 dataset.

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

- Python 3.6 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

git clone https://github.com/Ankitb700/Food_Vision_Big.git

2. Navigate to the project directory:

3. Create a new virtual environment (optional but recommended):

python -m venv env

4. Activate the virtual environment:

   - On Windows:
   
     ```
     env\Scripts\activate
     ```
     
   - On Unix or Linux:
   
     ```
     source env/bin/activate
     ```
     
5. Install the required packages:

pip install -r requirements.txt


2. Open your web browser and navigate to `http://localhost:5000`.

3. Click the "Choose File" button and select an image from your local machine.

4. Click the "Submit" button to upload the image.

5. The predicted name of the food item will be displayed on the webpage.

### Dataset

This project uses the Food-101 dataset, which consists of 101 food categories with 101,000 images. The dataset was introduced by Ece Kaynar and Anıl Körpe from Purdue University in 2021. For more information about the dataset, please visit the [Food-101 website](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

### Model

The project utilizes a pre-trained deep learning model based on the PyTorch framework. The model was trained on the Food-101 dataset and can accurately classify images into one of the 101 food categories.

### Contributing

If you'd like to contribute to the project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive commit messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

### License

This project is licensed under the [MIT License](LICENSE).

### Acknowledgments

- The Food-101 dataset was introduced by Ece Kaynar and Anıl Körpe from Purdue University.
- The project utilizes the PyTorch deep learning framework.

