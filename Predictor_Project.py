# -*- coding: utf-8 -*-
"""
Placement Predictor — Model Training Script

STEPS:
  0. Preprocess + EDA + Feature Selection
  1. Extract input and output cols
  2. Scale the values
  3. Train/test split
  4. Train the model
  5. Evaluate the model
  6. Save model + scaler for deployment
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_and_save_model():
    # Load dataset (relative path)
    df = pd.read_csv('placement.csv')
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    # Drop the unnamed index column if present
    if df.columns[0] == '' or 'Unnamed' in df.columns[0]:
        df = df.iloc[:, 1:]

    print(f"\nFeatures: {list(df.columns)}")

    # Extract input (X) and output (y)
    X = df.iloc[:, 0:2]  # cgpa, iq
    y = df.iloc[:, -1]   # placement

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression model
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")

    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("[OK] Saved model.pkl")

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[OK] Saved scaler.pkl")

    return clf, scaler, acc


if __name__ == '__main__':
    train_and_save_model()
