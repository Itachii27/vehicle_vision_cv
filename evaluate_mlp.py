import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from mlp_model import RiskMLP

# Load and prepare data
df = pd.read_csv("data.csv")

# Encode labels
risk_encoder = LabelEncoder()
df['Risk Level'] = risk_encoder.fit_transform(df['Risk Level'])

# Features and labels
feature_cols = ["Box Area", "Center Offset", "Label Weight", "Distance Weight", "Risk Score"]
X = df[feature_cols].values
y = df["Risk Level"].values

# Load saved scaler
scaler = joblib.load("scaler.save")
X_scaled = scaler.transform(X)

# Train-test split (same as before to keep evaluation valid)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert test set to tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.astype("int64"), dtype=torch.long)

# Load model and weights
model = RiskMLP(input_size=5)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

# Evaluation
print("Accuracy:", accuracy_score(y_test, predicted))
print("Classification Report:\n", classification_report(y_test, predicted, target_names=risk_encoder.classes_))
