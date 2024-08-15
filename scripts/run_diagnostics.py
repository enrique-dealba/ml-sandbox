import hydra
from omegaconf import DictConfig

from utils.diagnostics import run_diagnostics


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    run_diagnostics(cfg)


if __name__ == "__main__":
    main()
