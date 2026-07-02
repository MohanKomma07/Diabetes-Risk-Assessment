"""
DIABETES RISK ASSESSMENT
==========================
Level: Basic to Intermediate
Goal : Predict whether a patient is at risk of diabetes (1) or not (0)
       based on health measurements.

NOTES ON DATA
-------------
The classic dataset for this project is the "Pima Indians Diabetes
Dataset", which contains 8 medical predictor features:
    Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
    BMI, DiabetesPedigreeFunction, Age
and a binary target 'Outcome' (1 = diabetic, 0 = not diabetic).

To keep this script self-contained and runnable offline, we GENERATE
a synthetic dataset with the same feature names and realistic value
ranges. If you have the real diabetes.csv file, simply replace the
"Load / Generate Data" section with:
    df = pd.read_csv("diabetes.csv")
and the rest of the pipeline works unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------
# 1. LOAD / GENERATE DATA
# -----------------------------------------------------------------
rng = np.random.default_rng(42)
n_samples = 1500

# Simulate each feature with a realistic distribution.
pregnancies = rng.integers(0, 15, n_samples)
glucose = rng.normal(120, 30, n_samples).clip(50, 250)
blood_pressure = rng.normal(70, 12, n_samples).clip(40, 130)
skin_thickness = rng.normal(20, 10, n_samples).clip(0, 60)
insulin = rng.normal(80, 60, n_samples).clip(0, 400)
bmi = rng.normal(31, 7, n_samples).clip(15, 55)
diabetes_pedigree = rng.uniform(0.05, 2.0, n_samples)
age = rng.integers(21, 80, n_samples)

# Build a "risk score" from the features so the target label is
# realistically correlated with the inputs (mimics real biology,
# where high glucose/BMI/age raise diabetes risk).
risk_score = (
    0.035 * glucose
    + 0.04 * bmi
    + 0.02 * age
    + 0.5 * diabetes_pedigree
    + 0.01 * insulin
    - 6.5
)
probability = 1 / (1 + np.exp(-risk_score))   # sigmoid -> probability of diabetes
outcome = (rng.uniform(0, 1, n_samples) < probability).astype(int)

df = pd.DataFrame({
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": diabetes_pedigree,
    "Age": age,
    "Outcome": outcome,
})

print("Dataset shape:", df.shape)
print("Diabetic cases:", df["Outcome"].sum(), "out of", len(df))
print(df.describe().round(1))

# -----------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# -----------------------------------------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------------
# 3. FEATURE SCALING (helps Logistic Regression converge well)
# -----------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------
# 4. MODEL 1 - LOGISTIC REGRESSION (simple, interpretable baseline)
# -----------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
log_reg_preds = log_reg.predict(X_test_scaled)

print("\n===== Logistic Regression Results =====")
print("Accuracy:", round(accuracy_score(y_test, log_reg_preds), 3))
print(classification_report(y_test, log_reg_preds, digits=3))

# -----------------------------------------------------------------
# 5. MODEL 2 - DECISION TREE (easy to visualize / explain to doctors)
# -----------------------------------------------------------------
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)

print("\n===== Decision Tree Results =====")
print("Accuracy:", round(accuracy_score(y_test, tree_preds), 3))
print(classification_report(y_test, tree_preds, digits=3))

# -----------------------------------------------------------------
# 6. MODEL 3 - RANDOM FOREST (usually the strongest of the three)
# -----------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n===== Random Forest Results =====")
print("Accuracy:", round(accuracy_score(y_test, rf_preds), 3))
print(classification_report(y_test, rf_preds, digits=3))

# 5-fold cross-validation gives a more reliable estimate of performance
# than a single train/test split.
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-validated accuracy (5-fold):", round(cv_scores.mean(), 3),
      "+/-", round(cv_scores.std(), 3))

# -----------------------------------------------------------------
# 7. CONFUSION MATRIX
# -----------------------------------------------------------------
cm = confusion_matrix(y_test, rf_preds)
print("\nConfusion Matrix (Random Forest):")
print("                 Predicted Low Risk  Predicted High Risk")
print(f"Actual Low Risk        {cm[0][0]:<18} {cm[0][1]}")
print(f"Actual High Risk       {cm[1][0]:<18} {cm[1][1]}")

# -----------------------------------------------------------------
# 8. FEATURE IMPORTANCE (which health factors matter most?)
# -----------------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
print("\nFeature importance ranking:")
print(importances.sort_values(ascending=False))

# -----------------------------------------------------------------
# 9. SIMPLE PREDICTION FOR A NEW PATIENT
# -----------------------------------------------------------------
new_patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 150,
    "BloodPressure": 80,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 33.5,
    "DiabetesPedigreeFunction": 0.6,
    "Age": 45,
}])
risk_prediction = rf_model.predict(new_patient)[0]
risk_probability = rf_model.predict_proba(new_patient)[0][1]
print(f"\nNew patient prediction: {'HIGH RISK' if risk_prediction == 1 else 'LOW RISK'}"
      f" (probability of diabetes: {risk_probability:.2%})")

# -----------------------------------------------------------------
# 10. NEXT STEPS (ideas to extend this project)
# -----------------------------------------------------------------
# - Handle missing/zero values properly (in the real Pima dataset,
#   0s in Glucose/BMI/BloodPressure actually mean "missing").
# - Try hyperparameter tuning with GridSearchCV.
# - Build a simple web form (e.g., Streamlit) so a user can input
#   their own health data and get a live risk prediction.
