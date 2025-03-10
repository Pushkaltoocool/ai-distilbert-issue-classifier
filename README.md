## Issue Classifier Using DistilBERT
This repository provides a complete pipeline for training a text classification model that categorizes incident reports into seven distinct issue types using HuggingFace's DistilBERT. The model is designed to classify descriptions such as infrastructure problems and other local issues based on text inputs.

## Overview
The project demonstrates how to:

Preprocess and tokenize a custom dataset of incident descriptions.
Map text data to seven predefined issue categories:
- Pothole
- Street Light Outage
- Graffiti
- Abandoned Vehicle
- Illegal Dumping
- Noise Complaint
- Other

## Packages needed
Perform a stratified train-test split to maintain a balanced category distribution.
Fine-tune a pretrained DistilBERT model for sequence classification.
Save the trained model and tokenizer for later inference.
Run an interactive testing loop to classify new descriptions with confidence thresholds.
Dependencies
Ensure you have Python 3.7+ installed and install the following packages:
- transformers
- datasets
- scikit-learn
- torch
Install these dependencies via pip:
pip install transformers datasets scikit-learn torch

## Data
The dataset is defined within the script and consists of 140 text descriptions, with 20 examples per each of the following seven categories:

- Pothole
- Street Light Outage
- Graffiti
- Abandoned Vehicle
- Illegal Dumping
- Noise Complaint
- Other
Each description represents a real-world incident example, written in an informal style.

## Project Structure
## Data Preparation
Dataset Definition: The script contains hard-coded incident descriptions and their corresponding category labels.
Stratified Split: The data is split into training and evaluation sets to preserve balanced category distributions.
Tokenization and Dataset Creation
Tokenization: Uses DistilBertTokenizer with truncation and padding.
Dataset Conversion: Converts tokenized data into HuggingFace Dataset objects for training and evaluation.
## Model Setup and Training
Model Initialization: Loads a pretrained DistilBertForSequenceClassification with a classification head adjusted for seven labels.
Training Arguments: Custom training parameters include:
10 training epochs
Batch size of 4 for both training and evaluation
A learning rate of 5e-5 and weight decay of 0.01
Warmup steps and logging, evaluation, and saving configured to run at the end of each epoch
Training Process: The model is fine-tuned using HuggingFace's Trainer API.
Saving and Inference
Model Saving: The trained model and tokenizer are saved locally for reuse.
Classification Functions:
classify_issue(description): Classifies a description and returns the predicted category along with its confidence.
classify_with_threshold(description, threshold=0.8): Flags low-confidence predictions for manual review.
Interactive Testing: A function is provided to interactively input text descriptions and observe the classification results.
## How to Run
Training the Model:
Run the script to begin the training process. The dataset will be tokenized, the model fine-tuned, and both the model and tokenizer will be saved in the ./issue_classifier_model directory.
Interactive Testing:
After training, the script enters an interactive testing loop. Type a description to get its assigned category and confidence score.
Type stop to exit the interactive testing.

## Inference Details
classify_issue(description):
Tokenizes the input description, performs inference using the fine-tuned model, and returns the predicted issue category along with a confidence score.

## classify_with_threshold(description, threshold=0.8):
Uses the basic classification function but flags predictions with a confidence below the specified threshold as "Uncertain" for manual review.

## Customization
Training Parameters:
You can adjust hyperparameters such as num_train_epochs, learning_rate, and batch sizes in the TrainingArguments section to better suit your dataset or address overfitting concerns.

## Dataset:
Currently, the dataset is embedded within the script. For larger or externally sourced datasets, modify the data loading and preprocessing sections accordingly.

## Confidence Threshold:
Modify the threshold in the classify_with_threshold function to set a desired level of confidence for automatic classification.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
HuggingFace Transformers
HuggingFace Datasets
PyTorch
