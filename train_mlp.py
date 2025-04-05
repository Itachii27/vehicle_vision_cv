import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlp_model import RiskMLP

# Load the data
df = pd.read_csv("risk_data.csv")

# Encode labels
risk_encoder = LabelEncoder()
df['risk_level'] = risk_encoder.fit_transform(df['risk_level'])  # Low=1, Medium=2, High=0 (depends on mapping)

# Features and labels
X = df[["box_area", "center_offset", "label_weight", "distance_weight", "risk_score"]].values
y = df["risk_level"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Model
model = RiskMLP(input_size=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# Save model and scaler
torch.save(model.state_dict(), "mlp_model.pth")
import joblib
joblib.dump(scaler, "scaler.save")
print("Training complete and model saved.")
