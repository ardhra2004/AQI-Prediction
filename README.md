# AQI Prediction
**Regression and Classification based Air Quality Index (AQI) prediction system using Pollutant data of various cities. The data set consists of Pollutant levels of 12 pollutants to estimate overall AQI values and classify air quality levels.**

## Overview
This project is a **Machine Learning and Web Application** developed using **Django** and **Python**, designed to predict and classify **Air Quality Index (AQI)** levels for Indian cities. It combines **Regression** and **Classification** models to forecast AQI values and categories (Good–Severe) based on key air pollutant concentrations.

The system provides a **user-friendly web dashboard** for data input, AQI prediction, and visualization of trends, model comparisons, and feature importance.

## Objectives
- Predict AQI values using regression models (e.g., Random Forest, XGBoost).  
- Classify AQI categories using classification algorithms.  
- Develop a web interface to visualize and interpret predictions.  
- Improve awareness and assist policymakers in air quality management.
## System Architecture
The project follows a **3-layer modular architecture**:

1. **Data Layer:**  
   Handles pollutant data ingestion, cleaning, and storage.  
2. **Processing & Modeling Layer:**  
   Performs feature engineering, regression/classification training, and evaluation.  
3. **Presentation Layer:**  
   Django web interface displaying AQI predictions, trends, and performance metrics.

## Evaluation metrics
- For regression:
R², RMSE, MAE

- For classification:
Accuracy, precision, recall, F1, confusion matrix, ROC-AUC 

## Model Performance
| Model         | Type           | Metric   | Value |
| ------------- | -------------- | -------- | ----- |
| Random Forest | Regression     | R²       | 0.91  |
| Random Forest | Classification | Accuracy | 0.89  |

## Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| Language | Python 3.10+ |
| Framework | Django |
| ML Libraries | scikit-learn, pandas, numpy, xgboost |
| Visualization | matplotlib, seaborn, Chart.js |
| Web | HTML5, CSS3, Bootstrap |

## Future work

- Add meteorological features (temperature, humidity, wind speed), which strongly influence AQI.
- Time-series models for forecasting multi-step AQI.
- Real-time ingestion from public APIs or streaming pipeline.
- Deploy SHAP explanations with caching for interactive UI.

## References

Gupta, N. S., et al. (2023). Prediction of Air Quality Index using Machine Learning Techniques: A Comparative Analysis. Journal of Environmental and Public Health. https://doi.org/10.1155/2023/4916267

Kaggle Dataset — Air Quality Data in India (2015–2020). https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
