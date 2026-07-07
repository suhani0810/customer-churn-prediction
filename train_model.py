"""
Customer Churn Prediction - Model Training
Dataset: IBM Telco Customer Churn
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pickle

# ---- Load & clean ----
df = pd.read_csv("telco_churn.csv")
df.drop("customerID", axis=1, inplace=True)

# TotalCharges has blank strings for new customers -> convert & fill
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# ---- Encode categoricals ----
encoders = {}
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("Churn")

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_le = LabelEncoder()
df["Churn"] = target_le.fit_transform(df["Churn"])  # No=0, Yes=1

# ---- Train/test split ----
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Train model ----
model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# ---- Evaluate ----
preds = model.predict(X_test)
print("Accuracy :", round(accuracy_score(y_test, preds), 4))
print("Precision:", round(precision_score(y_test, preds), 4))
print("Recall   :", round(recall_score(y_test, preds), 4))
print("F1 Score :", round(f1_score(y_test, preds), 4))
print("\n", classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

# ---- Feature importance ----
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 drivers of churn:\n", importances.head(5))

# ---- Save model + encoders + column order ----
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "encoders": encoders,
        "target_le": target_le,
        "columns": X.columns.tolist()
    }, f)

print("\nSaved model.pkl")
