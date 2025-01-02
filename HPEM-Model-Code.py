#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets (replace with actual dataset paths or loading code)
# Example datasets: Replace with real data loading
heart_data = pd.read_csv('heart.csv')
cardio_data = pd.read_csv('cardio.csv')
alt_heart_data = pd.read_csv('alt_heart.csv')

def preprocess_data(data, target_column):
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Handle missing values (mean for numerical, mode for categorical)
    X.fillna(X.mean(), inplace=True)

    # Normalize numerical features
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = (X[col] - X[col].min()) / (X[col].max() - X[col].min())

    return X, y

def train_hpem_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection using mutual information
    k = 15  # Number of top features to select
    mutual_info = mutual_info_classif(X_train, y_train)
    top_k_indices = np.argsort(mutual_info)[-k:]
    X_train = X_train.iloc[:, top_k_indices]
    X_test = X_test.iloc[:, top_k_indices]

    # Initialize base learners
    rf = RandomForestClassifier(random_state=42)
    gbm = GradientBoostingClassifier(random_state=42)
    svm = SVC(probability=True, random_state=42)

    # Train base learners
    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    # Soft voting ensemble
    rf_preds = rf.predict_proba(X_train)[:, 1]
    gbm_preds = gbm.predict_proba(X_train)[:, 1]
    svm_preds = svm.predict_proba(X_train)[:, 1]

    ensemble_preds = np.mean([rf_preds, gbm_preds, svm_preds], axis=0)

    # Train meta-learner (XGBoost)
    meta_learner = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    meta_learner.fit(ensemble_preds.reshape(-1, 1), y_train)

    # Test set predictions
    rf_test_preds = rf.predict_proba(X_test)[:, 1]
    gbm_test_preds = gbm.predict_proba(X_test)[:, 1]
    svm_test_preds = svm.predict_proba(X_test)[:, 1]

    test_ensemble_preds = np.mean([rf_test_preds, gbm_test_preds, svm_test_preds], axis=0)
    final_preds = meta_learner.predict(test_ensemble_preds.reshape(-1, 1))

    # Evaluate performance
    accuracy = accuracy_score(y_test, final_preds)
    precision = precision_score(y_test, final_preds)
    recall = recall_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)

    return accuracy, precision, recall, f1

# Preprocess and train on each dataset
datasets = [(heart_data, 'target'), (cardio_data, 'target'), (alt_heart_data, 'target')]
for data, target in datasets:
    X, y = preprocess_data(data, target)
    accuracy, precision, recall, f1 = train_hpem_model(X, y)
    print(f"Dataset: {target}\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-Score: {f1}\n")

