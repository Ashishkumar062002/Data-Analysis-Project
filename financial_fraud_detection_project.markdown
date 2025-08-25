# Advanced Financial Fraud Detection Project

## Project Overview
This project develops a sophisticated financial fraud detection system using machine learning techniques to identify fraudulent transactions in a financial dataset. It leverages advanced data preprocessing, feature engineering, and ensemble modeling to achieve high accuracy and recall, addressing the challenges of imbalanced datasets common in fraud detection. The project includes a Python implementation with visualization and a Flask API for potential real-time deployment.

## Objectives
- Detect fraudulent financial transactions with high accuracy and minimal false positives
- Implement robust feature engineering to capture behavioral patterns
- Handle imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)
- Provide actionable insights through visualizations
- Create a scalable framework for real-time fraud detection

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- imbalanced-learn
- xgboost
- matplotlib
- seaborn
- flask

## Installation
1. Install required packages:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn flask
```

## Project Structure
- `fraud_detection.py`: Main script for data preprocessing, model training, and evaluation
- `requirements.txt`: List of dependencies
- `fraud_detection_api.py`: Flask API for real-time predictions
- `README.md`: Project documentation
- `data/`: Directory for dataset (synthetic data used in this example)

## Methodology
1. **Data Acquisition**: Uses a synthetic financial dataset with features like transaction amount, account balances, and transaction type.
2. **Preprocessing**: Handles missing values, normalizes numerical features, and encodes categorical variables.
3. **Feature Engineering**: Creates time-based features, transaction amount categories, and balance change ratios.
4. **Model Selection**: Implements an ensemble approach with Random Forest and XGBoost classifiers.
5. **Handling Imbalance**: Applies SMOTE to address class imbalance in fraud data.
6. **Evaluation**: Uses metrics like ROC AUC, precision, recall, and confusion matrix.
7. **Visualization**: Generates plots for feature importance, ROC curves, and confusion matrices.
8. **API**: Provides a Flask API for real-time transaction analysis.

## Code Implementation
Below is the main Python script for the fraud detection system:

<xaiArtifact artifact_id="a4d21d80-0000-4d6c-9ba9-0a01554106ab" artifact_version_id="7954ab9f-250f-4a2a-8988-f69cd22c7b5e" title="fraud_detection.py" contentType="text/python">

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Generate synthetic dataset (replace with real dataset if available)
def generate_synthetic_data(n_samples=10000):
    np.random.seed(42)
    data = {
        'step': np.random.randint(1, 744, n_samples),
        'type': np.random.choice(['CASH_IN', 'CASH_OUT', 'TRANSFER', 'PAYMENT'], n_samples),
        'amount': np.random.lognormal(mean=10, sigma=1, size=n_samples),
        'oldbalanceOrg': np.random.lognormal(mean=10, sigma=1, size=n_samples),
        'newbalanceOrig': np.random.lognormal(mean=10, sigma=1, size=n_samples),
        'oldbalanceDest': np.random.lognormal(mean=10, sigma=1, size=n_samples),
        'newbalanceDest': np.random.lognormal(mean=10, sigma=1, size=n_samples),
        'isFraud': np.random.choice([0, 1], n_samples, p=[0.995, 0.005])
    }
    return pd.DataFrame(data)

# Feature engineering
def engineer_features(df):
    df['balanceChangeOrg'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balanceChangeDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['hour'] = df['step'] % 24
    df['day'] = df['step'] // 24
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    return df

# Main execution
def main():
    # Load or generate data
    df = generate_synthetic_data()
    
    # Feature engineering
    df = engineer_features(df)
    
    # Prepare features and target
    X = df.drop(['isFraud'], axis=1)
    y = df['isFraud']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                      'balanceChangeOrg', 'balanceChangeDest', 'amount_to_balance_ratio']
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Train models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBClassifier(random_state=42)
    
    rf_model.fit(X_train_balanced, y_train_balanced)
    xgb_model.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    models = {'Random Forest': rf_model, 'XGBoost': xgb_model}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # ROC Curve
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'{name} ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.show()
    
    # Feature Importance (XGBoost)
    feature_importance = xgb_model.feature_importances_
    feature_names = X.columns
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feature_importance, y=feature_names)
    plt.title('XGBoost Feature Importance')
    plt.show()

if __name__ == "__main__":
    main()