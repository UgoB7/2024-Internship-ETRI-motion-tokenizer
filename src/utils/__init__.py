import logging
import os
import warnings
from typing import List, Sequence

import hydra
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Applies optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Setting global seeds
    - Converting relative ckpt path to absolute path
    - Rich config printing
    """

    # disable python warnings
    if config.get("ignore_warnings"):
        log.info(f"Disabling python warnings! <config.ignore_warnings={config.ignore_warnings}>")
        warnings.filterwarnings("ignore")

    # set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        log.info(f"Setting seeds! <config.seed={config.seed}>")
        pl.seed_everything(config.seed, workers=True)

    # convert ckpt path to absolute path
    ckpt_path = config.get("ckpt_path")
    if ckpt_path and not os.path.isabs(ckpt_path):
        log.info(f"Converting ckpt path to absolute path! <config.ckpt_path={ckpt_path}>")
        config.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), ckpt_path)

    # pretty print config tree using Rich library
    if config.get("print_config"):
        log.info(f"Printing config tree with Rich! <config.print_config={config.print_config}>")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    for field in print_order:
        queue.append(field) if field in config else log.info(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    for field in config:
        if field not in queue:
            queue.append(field)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    hparams["datamodule"] = config["datamodule"]
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]
    if "ckpt_path" in config:
        hparams["ckpt_path"] = config.ckpt_path

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


def get_metric_value(metric_name: str, trainer: pl.Trainer) -> float:
    """Retrieves value of the metric logged in LightningModule.

    Args:
        metric_name (str): Name of the metric.
        trainer (pl.Trainer): Lightning Trainer instance.
    """
    if metric_name not in trainer.callback_metrics:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure the `optimized_metric` name in `hparams_search` config is correct!"
        )
    return trainer.callback_metrics[metric_name]


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
