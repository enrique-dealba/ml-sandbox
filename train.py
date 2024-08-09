import sys
import time

import click
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf

from models import BaseModel, ExperimentModel
from utils.data_loader import get_data_loaders


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# Redirect stdout and stderr
log_file = open("/app/logs/output.log", "w")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)


def log_message(message, color=None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if color:
        message = click.style(message, fg=color)
    print(f"{timestamp} - {message}")


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    log_message("Training started", color="green")

    # Log configuration information
    log_message("Configuration:", color="blue")
    log_message(f"Learning rate: {cfg.training.learning_rate}")
    log_message(f"Number of epochs: {cfg.training.epochs}")
    log_message(f"Batch size: {cfg.data.batch_size}")
    log_message(f"Model type: {cfg.model.type}")
    log_message(f"Device: {cfg.training.device}")

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    log_message(f"Full Wandb config: {wandb_config}", color="cyan")

    wandb.init(project="mnist-sandbox", config=wandb_config)

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}", color="yellow")

    train_loader, val_loader, test_loader = get_data_loaders(cfg)

    try:
        if cfg.model.type == "base":
            model = BaseModel(cfg).to(device)
        elif cfg.model.type == "experiment":
            model = ExperimentModel(cfg).to(device)
        else:
            raise ValueError(f"Unknown model type: {cfg.model.type}")
    except AttributeError as e:
        log_message(f"Configuration error: {e}", color="red")
        log_message(
            "Check your config files for missing or incorrect parameters.", color="red"
        )
        return

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    best_val_accuracy = 0
    patience = cfg.training.patience
    patience_counter = 0

    for epoch in range(cfg.training.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()

            if batch_idx % 100 == 0:
                log_message(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}",
                    color="magenta",
                )

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100.0 * train_correct / len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100.0 * val_correct / len(val_loader.dataset)

        # Log metrics to wandb
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        log_message(
            f"Epoch {epoch}: Train loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
            f"Val loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%",
            color="cyan",
        )

        # Save the best model and update best_val_accuracy
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
            artifact = wandb.Artifact(f"best_model_run_{wandb.run.id}", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

            log_message(f"Saved best model to {model_path}", color="green")
            log_message(
                f"New best validation accuracy: {best_val_accuracy:.2f}%", color="green"
            )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            log_message(
                f"Early stopping triggered after {epoch + 1} epochs", color="yellow"
            )
            break

    # Final evaluation on test set
    log_message("Running Final Evaluation on Test Set...", color="green")
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * test_correct / len(test_loader.dataset)

    # Log final test metrics
    wandb.log(
        {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "best_val_accuracy": best_val_accuracy,
        }
    )

    log_message(
        f"Final Test loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%",
        color="green",
    )
    log_message(f"Best Validation Accuracy: {best_val_accuracy:.2f}%", color="green")
    log_message(f"wandb Run ID: {wandb.run.id}", color="blue")
    log_message(f"wandb Run Name: {wandb.run.name}", color="blue")

    wandb.finish()


if __name__ == "__main__":
    log_message("Script starting...", color="green")
    train()
