# Municipal Solid Waste Classification Using VGG16

This project uses a pre-trained VGG16 deep learning model to classify images of municipal solid waste into three categories: **Biodegradable**, **Recyclable**, and **Trash**. The model is trained on a Kaggle dataset with data augmentation and early stopping to improve accuracy and prevent overfitting.

## Dataset

The dataset is sourced from [Kaggle - Municipal Solid Waste Dataset](https://www.kaggle.com/datasets/elinachen717/municipal-solid-waste-dataset), containing labeled images of different waste types.

## Features

- Dataset split into training, validation, and test sets
- Image augmentation for better generalization
- Transfer learning using VGG16
- Prediction on individual images
- Model saved for future use

## Usage

1. Download the dataset using Kaggle API.
2. Run the training notebook/script.
3. Use the saved model for inference on new images.
