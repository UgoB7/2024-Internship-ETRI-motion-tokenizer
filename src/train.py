from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

def inspect_batch_sizes(dataloader):
    for i, batch in enumerate(dataloader):
        if isinstance(batch, torch.Tensor):
            print(f"Batch {i} size: {batch.size()}")
        elif isinstance(batch, (list, tuple)):
            print(f"Batch {i} size: {[x.size() for x in batch if isinstance(x, torch.Tensor)]}")
        else:
            print(f"Batch {i} contains data of type: {type(batch)}")
        if i >= 5:  # Limit to a few batches for debugging
            break

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    try:
            
        """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
        training.

        This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
        failure. Useful for multiruns, saving info about the crash, etc.

        :param cfg: A DictConfig configuration composed by Hydra.
        :return: A tuple with metrics and dict with all instantiated objects.
        """
        # set seed for random number generators in pytorch, numpy and python.random
        if cfg.get("seed"):
            L.seed_everything(cfg.seed, workers=True)

        log.info(f"Instantiating datamodule <{cfg.data._target_}>")
        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

        # Inspect the batch sizes before training
        log.info("Inspecting batch sizes from train dataloader...")
        inspect_batch_sizes(datamodule.train_dataloader())

        # If needed, also inspect validation or test dataloaders:
        # log.info("Inspecting batch sizes from validation dataloader...")
        # inspect_batch_sizes(datamodule.val_dataloader())

        # log.info("Inspecting batch sizes from test dataloader...")
        # inspect_batch_sizes(datamodule.test_dataloader())

        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        #print(f"[DEBUG] Model: {model}")

        log.info("Instantiating callbacks...")
        callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
        #print(f"[DEBUG] Callbacks: {callbacks}")

        log.info("Instantiating loggers...")
        logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
        #print(f"[DEBUG] Loggers: {logger}")

        log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
        #print(f"[DEBUG] Trainer: {trainer}")

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "callbacks": callbacks,
            "logger": logger,
            "trainer": trainer,
        }

        if logger:
            log.info("Logging hyperparameters!")
            log_hyperparameters(object_dict)

        if cfg.get("train"):
            log.info("Starting training!")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

        train_metrics = trainer.callback_metrics
        print(f"[DEBUG] Training metrics: {train_metrics}")

        if cfg.get("test"):
            log.info("Starting testing!")
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                log.warning("Best ckpt not found! Using current weights for testing...")
                ckpt_path = None
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            log.info(f"Best ckpt path: {ckpt_path}")


        test_metrics = trainer.callback_metrics

        # merge train and test metrics
        metric_dict = {**train_metrics, **test_metrics}

        return metric_dict, object_dict

    except Exception as e:
            log.error(f"Exception during training: {e}")
            raise


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    try:
        extras(cfg)  # Apply extra utilities like printing the configuration

        # Train the model
        metric_dict, _ = train(cfg)

        # Retrieve metric value for hyperparameter optimization
        metric_value = get_metric_value(
            metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
        )

        return metric_value
    except Exception as e:
        log.error(f"Exception in main function: {e}")
        raise


if __name__ == "__main__":
    main()

