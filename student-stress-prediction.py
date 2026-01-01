# ==========================================
# Student Stress Prediction System
# (Google Colab – Copy & Paste Ready)
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

# ------------------------
# Load Dataset
# ------------------------
df = pd.read_csv("/media/venkat/DVS/Shyaam/python-projects/dataset.csv")

print("\nDataset Preview:")
print(df.head())

print("\nDataset Columns:")
print(df.columns)

# ------------------------
# Encode categorical columns
# ------------------------
gender_encoder = LabelEncoder()
df["gender"] = gender_encoder.fit_transform(df["gender"])

stress_type_encoder = LabelEncoder()
df["stress_type"] = stress_type_encoder.fit_transform(df["stress_type"])

# ------------------------
# Features & Targets
# ------------------------
X = df.drop(["stress_experience", "stress_type"], axis=1)

y_stress_experience = df["stress_experience"]   # Regression target
y_stress_type = df["stress_type"]               # Classification target

FEATURE_COLUMNS = X.columns.tolist()

# ------------------------
# Train-Test Split
# ------------------------
X_train, X_test, y_exp_train, y_exp_test, y_type_train, y_type_test = train_test_split(
    X,
    y_stress_experience,
    y_stress_type,
    test_size=0.2,
    random_state=42
)

# ------------------------
# Feature Scaling
# ------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------
# Train Models
# ------------------------
stress_exp_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
stress_exp_model.fit(X_train, y_exp_train)

stress_type_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
stress_type_model.fit(X_train, y_type_train)

# ------------------------
# Evaluation
# ------------------------
exp_pred = stress_exp_model.predict(X_test)
print("\nStress Experience MAE:", mean_absolute_error(y_exp_test, exp_pred))

type_pred = stress_type_model.predict(X_test)
print("\nStress Type Accuracy:", accuracy_score(y_type_test, type_pred))
print("\nStress Type Classification Report:")
print(classification_report(y_type_test, type_pred))

# ------------------------
# User Input Functions
# ------------------------
def get_numeric_input(feature):
    while True:
        try:
            return float(input(f"Enter {feature}: "))
        except ValueError:
            print("Please enter a numeric value.")

def get_user_input_dataframe(feature_columns):
    print("\n--- Enter Student Details ---")
    values = []
    for feature in feature_columns:
        values.append(get_numeric_input(feature))
    return pd.DataFrame([values], columns=feature_columns)
print("male=1")
print("female=0")

# ------------------------
# Custom Student Prediction
# ------------------------
user_df = get_user_input_dataframe(FEATURE_COLUMNS)

user_scaled = scaler.transform(user_df)

predicted_stress_experience = stress_exp_model.predict(user_scaled)[0]
predicted_stress_type_encoded = stress_type_model.predict(user_scaled)[0]

predicted_stress_type = stress_type_encoder.inverse_transform(
    [predicted_stress_type_encoded]
)[0]

print("\n--- Prediction Result ---")
print(f"Predicted Stress Experience Level: {predicted_stress_experience:.2f}")
print(f"Predicted Stress Type: {predicted_stress_type}")
