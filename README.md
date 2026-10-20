Loan Status Prediction Pipeline 🏦

A streamlined, end-to-end machine learning pipeline built in Python to predict loan approval statuses. This project automates the data cleaning, preprocessing, training, and export of a robust RandomForestClassifier.

🚀 Features

Automated Data Cleaning: Dynamically strips whitespace from column headers to prevent key errors.

Smart Target Detection: Automatically scans for the target column based on keywords (loan and status), making the script adaptable to slight dataset variations.

Robust Preprocessing: Handles missing values seamlessly using mode imputation and encodes all categorical text variables dynamically using LabelEncoder.

Model Serialization: Trains a Random Forest model and automatically exports it as a production-ready loan_model.pkl file using Pickle.

📁 Project Structure

├── model_training.py   # The main training pipeline script
├── train.csv           # Training dataset (needs to be placed in root)
├── loan_model.pkl      # Generated after training (Serialized ML model)
└── README.md           # Project documentation


🛠️ Prerequisites

Ensure you have Python 3.7+ installed along with the following libraries:

pip install pandas scikit-learn


💻 How to Use

Prepare your data: Ensure your dataset is named train.csv and placed in the same directory as the script. The dataset should contain a target column indicating the loan status (e.g., Loan_Status with 'Y' or 'N' values).

Run the pipeline: Execute the training script from your terminal:

python model_training.py


Review the output: The console will display the detected columns, the identified target column, and the final model accuracy.

Deploy: Once the script completes successfully, a loan_model.pkl file will be generated in your directory. This file can be loaded into a Flask, FastAPI, or Streamlit backend for real-time predictions.

📊 Pipeline Workflow

Loads the dataset.

Cleans column names and drops non-predictive IDs (Loan_ID).

Fills missing data points with the most frequent values (mode).

Auto-detects the target variable and maps approvals/rejections to 1 and 0.

Encodes all object-type features into numeric labels.

Splits data into 80% training and 20% testing sets.

Trains a Random Forest Classifier.

Evaluates accuracy and saves the model.
