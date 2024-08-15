# import neural_net_checklist.torch_diagnostics as torch_diagnostics
# import torch
# from omegaconf import OmegaConf

# from models import BaseModel, ExperimentModel
# from utils.data_loader import get_data_loaders


# def run_diagnostics(cfg):
#     device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

#     print(f"Using device: {device}")
#     print(f"CUDA available: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"Current CUDA device: {torch.cuda.current_device()}")
#         print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
#         print(f"CUDA device count: {torch.cuda.device_count()}")
#         print(f"CUDA version: {torch.version.cuda}")

#     # Get data loaders
#     train_loader, _, _ = get_data_loaders(cfg)

#     def print_model_config(model_cfg):
#         for key, value in model_cfg.items():
#             print(f"  {key}: {value}")

#     # Define model creation functions
#     def create_base_model():
#         print("Creating Base Model with config:")
#         print_model_config(cfg.model)
#         return BaseModel(cfg).to(device)

#     def create_experiment_model():
#         print("Creating Experiment Model with config:")
#         print_model_config(cfg.model)
#         return ExperimentModel(cfg).to(device)

#     # Run diagnostics for base model
#     print("Running diagnostics for Base Model:")
#     torch_diagnostics.assert_all_for_classification_cross_entropy_loss(
#         create_base_model, train_loader, device=device, num_classes=cfg.data.num_classes
#     )

#     # Run diagnostics for experiment model
#     print("\nRunning diagnostics for Experiment Model:")
#     torch_diagnostics.assert_all_for_classification_cross_entropy_loss(
#         create_experiment_model,
#         train_loader,
#         device=device,
#         num_classes=cfg.data.num_classes,
#     )

import neural_net_checklist.torch_diagnostics as torch_diagnostics
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
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

    def print_model_config(model_cfg):
        for key, value in model_cfg.items():
            if key != '_target_':
                print(f"  {key}: {value}")

    def create_model():
        print(f"Creating model with config:")
        print_model_config(cfg.experiment.model)
        model = instantiate(cfg.experiment.model)
        return model.to(device)

    # Run diagnostics for the specified model
    model_name = cfg.experiment.model._target_.split('.')[-1]
    print(f"Running diagnostics for {model_name}:")
    torch_diagnostics.assert_all_for_classification_cross_entropy_loss(
        create_model,
        train_loader,
        device=device,
        num_classes=cfg.data.num_classes
    )
