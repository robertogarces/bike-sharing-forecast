import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.paths.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    token_path = Path.home() / ".kaggle" / "access_token"
    token = token_path.read_text().strip()
    os.environ["KAGGLE_TOKEN"] = token

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    logger.info(f"Downloading dataset {cfg.dataset.kaggle_slug} to {raw_dir}")
    api.dataset_download_files(
        cfg.dataset.kaggle_slug,
        path=raw_dir,
        unzip=True,
    )
    logger.info("Download complete")


if __name__ == "__main__":
    main()