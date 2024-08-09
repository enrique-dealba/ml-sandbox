import torch
from torchvision import datasets, transforms

def get_mnist_subset(subset_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    subset = torch.utils.data.Subset(full_dataset, subset_indices)
    return subset

def get_data_loaders(config):
    train_dataset = get_mnist_subset(config.data.subset_size)
    test_dataset = datasets.MNIST('data', train=False, download=True, 
                                  transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False)
    
    return train_loader, test_loader
