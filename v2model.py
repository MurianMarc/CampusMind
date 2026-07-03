import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import json

with open("config.json", "r") as file:
    config = json.load(file)


BASE_DIR = Path(__file__).resolve().parent # folder your notebook/script is currently running in
training_csv = "data.csv"

csv_path = (BASE_DIR / training_csv).resolve()

df = pd.read_csv(csv_path)

features = [
    "attendance_percent_norm",
    "ca_score_norm",
    "participation_score_0_5",
    "lms_logins"
]

risk_inputs = pd.DataFrame({
    "attendance": 1 - df["attendance_percent_norm"],
    "ca": 1 - df["ca_score_norm"],
    "participation": (1 - df["participation_score_0_5"] / 5),
    "lms": 1 - (df["lms_logins"] / df["lms_logins"].max()).clip(upper=1) 
})

weights = pd.Series(config["feature_weights"])
weights = weights / weights.sum()  # Normalize weights to sum to 1

df["policy_risk_score"] = risk_inputs.dot(weights)

X = df[features]
y = df["risk_numeric"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


gb_model = GradientBoostingClassifier(random_state=42)

class_weights= {int(k): v for k, v in config["class_weights"].items()}
sample_weights = y_train.map(class_weights)

gb_model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = gb_model.predict(X_test)

print("Accuracy:" , accuracy_score(y_test, y_pred))
print("Precision:" , precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score:" , f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
