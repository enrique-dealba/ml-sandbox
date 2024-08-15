import neural_net_checklist.torch_diagnostics as torch_diagnostics
import torch
from omegaconf import OmegaConf

from models import BaseModel, ExperimentModel
from utils.data_loader import get_data_loaders


def run_diagnostics(cfg):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")

    # Get data loaders
    train_loader, _, _ = get_data_loaders(cfg)

    # Define model creation functions
    def create_base_model():
        return BaseModel(cfg).to(device)

    def create_experiment_model():
        return ExperimentModel(cfg).to(device)

    # Run diagnostics for base model
    print("Running diagnostics for Base Model:")
    torch_diagnostics.assert_all_for_classification_cross_entropy_loss(
        create_base_model, train_loader, device=device, num_classes=cfg.data.num_classes
    )

    # Run diagnostics for experiment model
    print("\nRunning diagnostics for Experiment Model:")
    torch_diagnostics.assert_all_for_classification_cross_entropy_loss(
        create_experiment_model,
        train_loader,
        device=device,
        num_classes=cfg.data.num_classes,
    )
