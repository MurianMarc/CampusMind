import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

X = df[features]
y = df["risk_numeric"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": Pipeline([
        ('scaler', StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
        ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "KNN":Pipeline([
        ('scaler', StandardScaler()),
        ("model", KNeighborsClassifier())
        ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ("model", SVC(probability=True,random_state=42))
        ])
}

results = []

for name,model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })
    
results_df = pd.DataFrame(results)

print(results_df)