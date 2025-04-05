import torch.nn as nn

class RiskMLP(nn.Module):
    def __init__(self, input_size):
        super(RiskMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)  # 3 classes: Low, Medium, High

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out
