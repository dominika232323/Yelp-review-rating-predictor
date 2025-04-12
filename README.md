# Yelp-review-rating-predictor

This repository contains a simple pipeline for training a model on a [dataset containing Yelp reviews](https://huggingface.co/datasets/Yelp/yelp_review_full).

## Getting Started

### Installing Dependencies

Ensure you have Python installed, then install the required libraries:

`pip install -r requirements.txt`

## Training the Model

Run the Jupyter Notebook train_and_evaluate.ipynb, which:

* Loads the dataset
* Imports the model from review_classifier.py
* Trains the model
* Saves metrics into Weights and Biases service
* Evaluates accuracy on the test set

After training, the model's accuracy on the Yelp reviews test set will be displayed in the notebook output.
