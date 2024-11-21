# Credit Card Default Prediction Model
## Project Overview
This project aims to predict whether a customer will default on their credit card payment. The dataset used contains various customer features, and the model is designed to classify customers into two categories: those who will default (1) and those who will not (0). The analysis and model training are performed using a Random Forest Classifier.

The main analysis is carried out in the Jupyter notebook classification.ipynb, which includes the following key steps:

# Steps:
## 1. Data Loading and Exploration:
The credit-g.arff dataset is loaded using the scipy.io.arff library.
The first few rows of the dataset are displayed for initial inspection.
The dataset is preprocessed by encoding the target variable class (default/no default) using binary values (0 for no default, 1 for default).
The dataset is then split into features (X) and labels (y) for model training.
## 2. Data Preprocessing:
Missing values in the dataset are identified and handled using median imputation for numerical features.
Data types are checked and categorical variables are encoded using LabelEncoder from sklearn.
## 3. Data Splitting:
The dataset is split into training and testing sets (80% for training and 20% for testing) using train_test_split from sklearn.
## 4. Model Training:
A Random Forest Classifier model is trained on the training dataset.
The model’s hyperparameters are set, including the number of estimators (trees) and a fixed random seed to ensure reproducibility.
## 5. Model Evaluation:
The model’s performance is evaluated using the testing dataset.
Key evaluation metrics are calculated:
Accuracy
Precision
Recall
F1-Score
A confusion matrix is generated to visualize the performance.
The ROC curve is plotted, and the AUC (Area Under Curve) score is calculated to evaluate the model's ability to distinguish between the two classes (default/no default).
## 6. Prediction:
The trained model can then be used to predict whether new or unseen data will result in a default.
Dependencies
To run this notebook, you will need to install the following dependencies:

pandas – for data manipulation
numpy – for numerical operations
matplotlib – for creating plots and visualizations
scikit-learn – for machine learning models and evaluation metrics
scipy – for handling ARFF file format
You can install the necessary dependencies by running:

bash
Copy code
pip install -r requirements.txt
