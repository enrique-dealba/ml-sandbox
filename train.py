import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from models import BaseModel, ExperimentModel
from utils.data_loader import get_data_loaders

@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    # Convert the Hydra config to a dictionary for wandb
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    
    wandb.init(project="mnist-sandbox", config=wandb_config)
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader = get_data_loaders(cfg)
    
    if cfg.model.type == "base":
        model = BaseModel(cfg).to(device)
    elif cfg.model.type == "experiment":
        model = ExperimentModel(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch})
        
        model.eval()
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
        wandb.log({"test_loss": test_loss, "accuracy": accuracy, "epoch": epoch})
        
        print(f'Epoch {epoch}: Test loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    torch.save(model.state_dict(), "model.pth")
    wandb.save("model.pth")

if __name__ == "__main__":
    train()
