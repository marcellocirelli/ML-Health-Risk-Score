# AI Optimization – Health Risk Score Prediction

This project focuses on building and comparing machine learning regression models to predict a **health risk score** based on structured health-related data.

The project emphasizes model selection, evaluation, and optimization rather than deep feature engineering or UI development.

## Overview

The goal of this project is to predict a continuous health risk score using supervised machine learning techniques.  
Multiple regression models were trained and evaluated to determine which approach produced the most accurate and stable results.

The final implementation includes ensemble-based models and visual analysis of predictions.

## Models Implemented

- **Random Forest Regressor**
- **Gradient Boosting Regressor**

Both models were trained on the same dataset and evaluated using standard regression metrics to compare performance.

## Features

- Data loading and preprocessing
- Model training and evaluation
- Performance comparison between ensemble models
- Visualization of predicted vs. actual values using **Matplotlib**
- Clear separation between data handling, modeling, and visualization logic

## Technologies Used

- **Python**
- **scikit-learn**
- **NumPy**
- **Matplotlib**
- Ensemble learning techniques (Random Forest, Gradient Boosting)

## Evaluation

Model performance was assessed using regression metrics such as:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Visual comparison of predictions vs. ground truth

Graphs were generated to support interpretability and model comparison.

## Purpose

This project demonstrates:
- Practical application of ensemble regression models
- Model comparison and optimization techniques
- Quantitative evaluation of predictive performance
- Use of visualization to interpret model behavior

## Notes

This project was developed and intended as an academic demonstration of applied machine learning concepts rather than a production healthcare system.
