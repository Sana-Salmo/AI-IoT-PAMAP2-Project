# AI Model in the IoT: PCA-Based Feature Reduction for IoT Sensor Data

This project demonstrates how Principal Component Analysis (PCA) can be used to reduce the dimensionality of sensor data for IoT applications, using simulated data similar to the PAMAP2 dataset.

## Features

- Simulates a dataset with 54 features (similar to PAMAP2).
- Introduces missing values and handles them using mean imputation.
- Applies PCA to retain 95% of the variance.
- Trains and evaluates five machine learning models:
  - SVM
  - Decision Tree
  - Random Forest
  - KNN
  - Logistic Regression
- Compares accuracy before and after PCA.

## Files Included

- `pca_iot_models.py`: Python script for the full experiment
- `README.md`: This file explaining the project

## How to Run

Install the required libraries:
```
pip install numpy pandas scikit-learn
```

Then run the script:
```
python pca_iot_models.py
```

## Authors

Rama Alkusair, Sana Rahmani  
Effat University – Spring 2025 – CS3081
