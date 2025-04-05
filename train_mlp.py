from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlp_model import RiskMLP
import joblib

# Load the data
df = pd.read_csv("data.csv")

# Keep track of loss & accuracy
losses = []
accuracies = []

# Encode 'Risk Level'
risk_encoder = LabelEncoder()
df['Risk Level'] = risk_encoder.fit_transform(df['Risk Level'])  # Low=1, Medium=2, High=0 (order may vary)

# Feature columns as per original CSV
feature_cols = ["Box Area", "Center Offset", "Label Weight", "Distance Weight", "Risk Score"]
X = df[feature_cols].values
y = df["Risk Level"].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.astype("int64"), dtype=torch.long)  # Fix: cast to int64 before passing to tensor

# Model setup
model = RiskMLP(input_size=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Track loss
    losses.append(loss.item())

    # Track train accuracy
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_train.numpy(), predicted.numpy())
    accuracies.append(acc)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f} | Accuracy: {acc:.4f}")

# Save model and scaler
torch.save(model.state_dict(), "mlp_model.pth")
joblib.dump(scaler, "scaler.save")
print("Training complete and model saved.")

# Plot loss and accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(accuracies, label='Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()