import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flatten = nn.Flatten()
        hidden_size = getattr(config.model, 'hidden_size', 128)  # Default to 128 if not specified
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
