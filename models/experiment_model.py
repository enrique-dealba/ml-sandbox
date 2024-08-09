import torch.nn as nn

class ExperimentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, config.model.hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.model.dropout)
        self.fc2 = nn.Linear(config.model.hidden_size, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
