# models/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle, json, os
from datetime import datetime

# Generate a unique version tag using the current timestamp (e.g., v20231024_153022)
VERSION_TAG = datetime.now().strftime("v%Y%m%d_%H%M%S")

# ── Load data (use any dataset you have) ─────────────────────────
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "osteoporosis.csv")
df = pd.read_csv(data_path)
cols_with_na = ['Alcohol Consumption', 'Medical Conditions', 'Medications']
for col in cols_with_na:
    df[col].fillna(df[col].mode()[0],inplace=True)


df['Calcium Intake'] = df['Calcium Intake'].map({'Low': 0, 'Adequate': 1, 'High': 2})
df['Physical Activity'] = df['Physical Activity'].map({'Sedentary': 0, 'Moderate': 1, 'Active': 2})


# 3. Handle Binary and Nominal Categorical Data
binary_cols = ['Gender', 'Hormonal Changes', 'Family History', 'Prior Fractures', 'Smoking']
multi_class_cols = ['Race/Ethnicity', 'Body Weight', 'Vitamin D Intake', 'Alcohol Consumption', 'Medical Conditions', 'Medications']

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
df = pd.get_dummies(df, columns=multi_class_cols, drop_first=True)

X = df.drop("Osteoporosis", axis=1)
y = df["Osteoporosis"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train['Age'] = scaler.fit_transform(X_train[['Age']])
X_test['Age'] = scaler.transform(X_test[['Age']])

top_features = [
    'Age', # (or 'Age_Group' if you binned it)

    'Prior Fractures', 
    'Family History', 
    'Gender',
    'Vitamin D Intake_Sufficient',
    'Physical Activity', # Replace with ordinal 'Vitamin D Intake' if you changed it
]
# Note: Make sure columns match your actual dataframe names after preprocessing
X_train = X_train[top_features]
X_test = X_test[top_features]

# ── Model A — XGBoost ─────────────────────────────────────────────
print("Training Model A (XGBoost)...")
model_a = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model_a.fit(X_train, y_train)

preds_a = model_a.predict(X_test)
proba_a = model_a.predict_proba(X_test)[:, 1]

metrics_a = {
    "accuracy":  round(accuracy_score(y_test, preds_a), 4),
    "roc_auc":   round(roc_auc_score(y_test, proba_a), 4),
    "model_type": "XGBoost",
    "features":   list(X.columns),
    "version":    VERSION_TAG
}

models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)
with open(os.path.join(models_dir, f"model_a_{VERSION_TAG}.pkl"), "wb") as f:
    pickle.dump(model_a, f)
with open(os.path.join(models_dir, f"model_a_{VERSION_TAG}_metadata.json"), "w") as f:
    json.dump(metrics_a, f, indent=2)

print(f"Model A — Accuracy: {metrics_a['accuracy']}, AUC: {metrics_a['roc_auc']}")

# ── Model B — LightGBM ────────────────────────────────────────────
print("Training Model B (LightGBM)...")
model_b = LGBMClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42,
    verbose=-1
)
model_b.fit(X_train, y_train)

preds_b = model_b.predict(X_test)
proba_b = model_b.predict_proba(X_test)[:, 1]

metrics_b = {
    "accuracy":   round(accuracy_score(y_test, preds_b), 4),
    "roc_auc":    round(roc_auc_score(y_test, proba_b), 4),
    "model_type": "LightGBM",
    "features":   list(X.columns),
    "version":    VERSION_TAG
}

with open(os.path.join(models_dir, f"model_b_{VERSION_TAG}.pkl"), "wb") as f:
    pickle.dump(model_b, f)
with open(os.path.join(models_dir, f"model_b_{VERSION_TAG}_metadata.json"), "w") as f:
    json.dump(metrics_b, f, indent=2)

print(f"Model B — Accuracy: {metrics_b['accuracy']}, AUC: {metrics_b['roc_auc']}")
print("\n✅ Both models trained and saved.")