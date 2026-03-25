import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

columns = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness",
    "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
]

# Pima Indians Diabetes Dataset — download from Kaggle and place as diabetes.csv
# OR run: pip install kaggle, then kaggle datasets download -d uciml/pima-indians-diabetes-database
csv_path = "diabetes.csv"

if os.path.exists(csv_path):
    print("Loading from local CSV...")
    df = pd.read_csv(csv_path)
    if "Outcome" not in df.columns:
        df.columns = columns
else:
    # Minimal embedded sample to verify pipeline works
    print("diabetes.csv not found — using embedded sample (download full dataset for better accuracy)")
    sample = [
        [6,148,72,35,0,33.6,0.627,50,1],[1,85,66,29,0,26.6,0.351,31,0],
        [8,183,64,0,0,23.3,0.672,32,1],[1,89,66,23,94,28.1,0.167,21,0],
        [0,137,40,35,168,43.1,2.288,33,1],[5,116,74,0,0,25.6,0.201,30,0],
        [3,78,50,32,88,31.0,0.248,26,1],[10,115,0,0,0,35.3,0.134,29,0],
        [2,197,70,45,543,30.5,0.158,53,1],[8,125,96,0,0,0.0,0.232,54,1],
        [4,110,92,0,0,37.6,0.191,30,0],[10,168,74,0,0,38.0,0.537,34,1],
        [10,139,80,0,0,27.1,1.441,57,0],[1,189,60,23,846,30.1,0.398,59,1],
        [5,166,72,19,175,25.8,0.587,51,1],[7,100,0,0,0,30.0,0.484,32,1],
        [0,118,84,47,230,45.8,0.551,31,1],[7,107,74,0,0,29.6,0.254,31,1],
        [1,103,30,38,83,43.3,0.183,33,0],[1,115,70,30,96,34.6,0.529,32,1],
        [3,126,88,41,235,39.3,0.704,27,0],[8,99,84,0,0,35.4,0.388,50,0],
        [7,196,90,0,0,39.8,0.451,41,1],[9,119,80,35,0,29.0,0.263,29,1],
        [11,143,94,33,146,36.6,0.254,51,1],[10,125,70,26,115,31.1,0.205,41,1],
        [7,147,76,0,0,39.4,0.257,43,1],[1,97,66,15,140,23.2,0.487,22,0],
        [13,145,82,19,110,22.2,0.245,57,0],[5,117,92,0,0,34.1,0.337,38,0],
        [5,109,75,26,0,36.0,0.546,60,0],[3,158,76,36,245,31.6,0.851,28,1],
        [3,88,58,11,54,24.8,0.267,22,0],[6,92,92,0,0,19.9,0.188,28,0],
        [10,122,78,31,0,27.6,0.512,45,0],[4,103,60,33,192,24.0,0.966,33,0],
        [11,138,76,0,0,33.2,0.420,35,0],[9,102,76,37,0,32.9,0.665,46,1],
        [2,90,68,42,0,38.2,0.503,27,1],[4,111,72,47,207,37.1,1.390,56,1],
        [3,180,64,25,70,34.0,0.271,26,0],[7,133,84,0,0,40.2,0.696,37,0],
        [7,106,92,18,0,22.7,0.235,48,0],[9,171,110,24,240,45.4,0.721,54,1],
        [7,159,64,0,0,27.4,0.294,40,0],[0,180,66,39,0,42.0,1.893,25,1],
        [1,146,56,0,0,29.7,0.564,29,0],[2,71,70,27,0,28.0,0.586,22,0],
        [7,103,66,32,0,39.1,0.344,31,1],[7,105,0,0,0,0.0,0.305,24,0],
        [1,103,80,11,82,19.4,0.491,22,0],[1,101,50,15,36,24.2,0.526,26,0],
        [5,88,66,21,23,24.4,0.342,30,0],[8,176,90,34,300,33.7,0.467,58,1],
        [7,150,66,42,342,34.7,0.718,42,0],[1,73,50,10,0,23.0,0.248,21,0],
        [7,187,68,39,304,37.7,0.254,41,1],[0,100,88,60,110,46.8,0.962,31,0],
        [0,146,82,0,0,40.5,1.781,44,0],[0,105,64,41,142,41.5,0.173,22,0],
        [2,84,0,0,0,0.0,0.304,21,0],[8,133,72,0,0,32.9,0.270,39,1],
        [5,44,62,0,0,25.0,0.587,36,0],[2,141,58,34,128,25.4,0.699,24,0],
        [7,114,66,0,0,32.8,0.258,42,1],[5,99,74,27,0,29.0,0.203,32,0],
        [0,109,88,30,0,32.5,0.855,38,1],[2,109,92,0,0,42.7,0.845,54,0],
        [1,95,66,13,38,19.6,0.334,25,0],[4,146,85,27,100,28.9,0.189,27,0],
        [8,105,100,36,0,43.3,0.239,45,1],[5,158,84,41,210,39.4,0.395,29,1],
        [1,105,58,0,0,24.3,0.187,21,0],[3,107,62,13,48,22.9,0.678,23,1],
        [4,109,64,44,99,34.6,0.955,26,0],[4,148,60,27,318,30.9,0.150,29,1],
        [0,113,80,16,0,31.0,0.874,21,0],[1,138,82,0,0,40.1,0.236,28,0],
        [0,108,68,20,0,27.3,0.787,32,0],[2,99,70,16,44,20.4,0.235,19,0],
        [6,103,72,32,190,37.7,0.324,55,0],[5,111,72,28,0,23.9,0.407,27,0],
        [8,196,76,29,280,37.5,0.605,57,1],[5,162,104,0,0,37.7,0.151,52,1],
        [1,96,64,27,87,33.2,0.289,21,0],[7,184,84,33,0,35.5,0.355,41,1],
        [2,81,60,22,0,27.7,0.290,25,0],[0,147,85,54,0,42.8,0.375,24,0],
        [7,179,95,31,0,34.2,0.164,60,0],[0,140,65,26,130,42.6,0.431,24,1],
        [9,112,82,24,0,28.2,1.282,50,1],[3,128,78,0,0,21.1,0.268,55,0],
        [4,127,88,11,155,34.5,0.598,28,0],[4,128,70,0,0,34.3,0.303,24,0],
        [1,100,66,15,56,23.6,0.666,26,0],[5,166,76,0,0,45.7,0.340,27,1],
        [3,94,96,31,0,36.7,0.254,23,0],[3,128,72,25,190,32.4,0.549,27,1],
        [0,137,84,27,0,27.3,0.231,59,0],[4,114,64,0,0,28.9,0.126,24,0],
        [2,100,54,28,105,37.8,0.498,24,0],[0,93,100,47,176,43.4,1.009,35,0],
        [4,97,60,23,0,28.2,0.443,22,0],[0,84,82,31,125,38.2,0.233,23,0],
        [4,120,68,0,0,29.6,0.709,34,0],[3,131,66,40,0,34.3,0.196,22,1],
    ]
    df = pd.DataFrame(sample, columns=columns)

print(f"Dataset shape: {df.shape}")
print(f"Outcome distribution:\n{df['Outcome'].value_counts()}")

# ── Preprocessing ─────────────────────────────────────────────────────────────
cols_with_zeros = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100, max_depth=8,
    min_samples_split=5, random_state=42, class_weight="balanced"
)
model.fit(X_train_scaled, y_train)

y_pred   = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["No Diabetes","Diabetes"]))

os.makedirs("../backend/model", exist_ok=True)
joblib.dump(model,          "../backend/model/diabetes_model.pkl")
joblib.dump(scaler,         "../backend/model/scaler.pkl")
joblib.dump(list(X.columns),"../backend/model/features.pkl")

print("\nModel saved to ../backend/model/")
print("Training complete!")
