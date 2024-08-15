import neural_net_checklist.torch_diagnostics as torch_diagnostics
import torch

from models import BaseModel, ExperimentModel
from utils.data_loader import get_data_loaders


def run_diagnostics(cfg):
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

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
