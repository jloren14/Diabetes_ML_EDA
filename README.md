# Diabetes Classification: Exploratory Data Analysis and Machine Learning 

This repository contains a Jupyter Notebook detailing the exploratory data analysis (EDA) and machine learning (ML) modeling performed on a diabetes dataset. The primary objective is to classify individuals as diabetic or non-diabetic based on various health-related attributes.

## Table of contents
* [Project Overview](#project-overview)
* [Dataset Information](#dataset-information)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Machine Learning Models](#machine-learning-models)
* [Results](#results)
* [Usage](#usage)
* [Dependencies](#dependencies)

## Project Overview

The goal of this project is to utilize EDA techniques to understand the underlying patterns in the data and to develop ML models that can accurately predict diabetes status. This involves data cleaning, visualization, feature selection, model training, and evaluation.

## Dataset Information

The dataset used in this project includes the following features:
* Age: Age of the patient
* Sex: Gender of the patient
* HighChol: Presence of high cholesterol
* CholCheck: Whether the patient had a cholesterol check
* BMI: Body Mass Index
* Smoker: Smoking status
* HeartDiseaseorAttack: History of heart disease or attack
* PhysActivity: Physical activity level
* Fruits: Fruit consumption frequency
* Veggies: Vegetable consumption frequency
* HvyAlcoholConsump: Heavy alcohol consumption
* GenHlth: General health status
* MentHlth: Number of days with poor mental health
* PhysHlth: Number of days with poor physical health
* DiffWalk: Difficulty in walking
* Stroke: History of stroke
* HighBP: Presence of high blood pressure
* Diabetes: Diabetes status (target variable)

## Exploratory Data Analysis

The EDA process includes:
* Data Inspection: Checking for null values, data types, and basic statistics.
* Data Cleaning: Renaming columns for clarity and mapping categorical variables to meaningful labels.
* Visualization: Plotting distributions and relationships between features and the target variable.

## Machine Learning Models

The following machine learning classifiers were explored:

* Logistic Regression
* Decision Trees
* Random Forest
* Support Vector Machines
* K-Nearest Neighbors
* Gradient Boosting Machines
* Naive Bayes
* Neural Networks
* XGBoost

Each model was evaluated using appropriate metrics to determine its performance in predicting diabetes status.

## Results

The models were assessed based on accuracy, precision, recall, and F1-score. Hyperparameter tuning was performed to optimize model performance. The top-performing model achieved an accuracy of `XX%`, indicating its effectiveness in classifying diabetes status.

## Usage

To replicate this analysis:

1. Clone the repository:
   ```bash
   git clone https://github.com/jloren14/Diabetes_ML_EDA.git
2. Navigate to the project directory:
   ``` bash
   cd Diabetes_ML_EDA
3. Install the required dependencies:
   ``` bash
   pip install -r requirements.txt
4. Launch Jupyter Notebook:
   ``` bash
   jupyter notebook
5. Open and run the `notebook.ipynb` to reproduce the analysis.

## Dependencies

The project requires the following Python packages:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* xgboost

These can be installed using the `requirements.txt` file provided.

