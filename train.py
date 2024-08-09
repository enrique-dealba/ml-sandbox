import logging
import sys

import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf

from models import BaseModel, ExperimentModel
from utils.data_loader import get_data_loaders


def setup_logger():
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler("training.log")

    # Create formatters and add it to handlers
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(format)
    f_handler.setFormatter(format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


logger = setup_logger()


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(project="mnist-sandbox", config=wandb_config)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    try:
        if cfg.model.type == "base":
            model = BaseModel(cfg).to(device)
        elif cfg.model.type == "experiment":
            model = ExperimentModel(cfg).to(device)
        else:
            raise ValueError(f"Unknown model type: {cfg.model.type}")
    except AttributeError as e:
        print(f"Configuration error: {e}")
        print("Check your config files for missing or incorrect parameters.")
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    best_val_accuracy = 0
    patience = cfg.training.patience
    patience_counter = 0

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
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * correct / len(val_loader.dataset)
        wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy, "epoch": epoch})

        print(
            f"Epoch {epoch}: Val loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
        )

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0

            # Save model
            model_path = f"best_model_epoch{epoch}_acc{val_accuracy:.2f}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "val_loss": val_loss,
                },
                model_path,
            )

            # Log model as artifact
            artifact = wandb.Artifact(f"best_model_run_{run.id}", type="model")
            artifact.add_file(model_path)
            run.log_artifact(artifact)

            print(f"Saved best model to {model_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    # Final evaluation on test set
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
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

    logger.info(
        f"Final Test loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%"
    )
    logger.info(f"wandb Run ID: {wandb.run.id}")
    logger.info(f"wandb Run Name: {wandb.run.name}")


if __name__ == "__main__":
    logger.info("Training starting...")
    train()
