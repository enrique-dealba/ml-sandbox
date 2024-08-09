import torch
from torchvision import datasets, transforms

def get_data_loaders(cfg):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    
    # Use the configured split ratio
    train_size = int(cfg.data.train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
