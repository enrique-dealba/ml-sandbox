import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from models import BaseModel, ExperimentModel
from utils.data_loader import get_data_loaders

@hydra.main(config_path="config", config_name="config")
def evaluate(cfg: DictConfig):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    _, test_loader = get_data_loaders(cfg)
    
    if cfg.model.type == "base":
        model = BaseModel(cfg).to(device)
    elif cfg.model.type == "experiment":
        model = ExperimentModel(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    evaluate()
