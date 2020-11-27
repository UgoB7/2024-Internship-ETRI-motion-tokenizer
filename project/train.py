from pytorch_lightning.profiler import SimpleProfiler
import yaml

# lightning modules
from lightning_modules.init_utils import *
from lightning_modules.callbacks import *


def train(project_config, run_config):

    # Init PyTorch Lightning model ⚡
    lit_model = init_lit_model(model_config=run_config["model"])

    # Init PyTorch Lightning datamodule ⚡
    datamodule = init_datamodule(dataset_config=run_config["dataset"])

    # Init Weights&Biases logger
    logger = init_wandb_logger(project_config, run_config, lit_model, datamodule)

    # Init checkpoint and early stoping callbacks
    callbacks = init_main_callbacks(project_config)

    # Add your custom callbacks
    callbacks.extend([
        # MetricsHeatmapLoggerCallback(),
        # UnfreezeModelCallback(wait_epochs=5),
        SaveCodeToWandbCallback(wandb_save_dir=logger.save_dir, lit_model=lit_model),
    ])

    # Init PyTorch Lightning trainer ⚡
    trainer = pl.Trainer(
        # whether to use gpu and how many
        gpus=project_config["num_of_gpus"],

        # experiment logging
        logger=logger,

        # useful callbacks
        callbacks=callbacks,

        # resuming training from checkpoint
        resume_from_checkpoint=project_config["resume_training"]["lightning_ckpt"]["ckpt_path"]
        if project_config["resume_training"]["lightning_ckpt"]["resume_from_ckpt"] else None,

        # print related
        progress_bar_refresh_rate=project_config["printing"]["progress_bar_refresh_rate"],
        profiler=SimpleProfiler() if project_config["printing"]["profiler"] else None,
        weights_summary=project_config["printing"]["weights_summary"],

        # run related
        max_epochs=run_config["trainer"]["max_epochs"],
        min_epochs=run_config["trainer"].get("min_epochs", 1),
        accumulate_grad_batches=run_config["trainer"].get("accumulate_grad_batches", 1),
        gradient_clip_val=run_config["trainer"].get("gradient_clip_val", 0.5),

        # these are mostly for debugging (read TIPS.md for explanation)
        fast_dev_run=run_config["trainer"].get("fast_dev_run", False),
        limit_train_batches=run_config["trainer"].get("limit_train_batches", 1.0),
        limit_val_batches=run_config["trainer"].get("limit_val_batches", 1.0),
        limit_test_batches=run_config["trainer"].get("limit_test_batches", 1.0),
        val_check_interval=run_config["trainer"].get("val_check_interval", 1.0),
        num_sanity_val_steps=run_config["trainer"].get("fast_dev_run", 3),

        # default log dir if no logger is found
        default_root_dir="logs/lightning_logs",
    )

    # Evaluate model on test set before training
    # trainer.test(model=lit_model, datamodule=datamodule)

    # Train the model
    trainer.fit(model=lit_model, datamodule=datamodule)

    # Evaluate model on test set after training
    trainer.test()


def load_config(path):
    with open(path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":

    # --------------------------------------- CHOOSE YOUR RUN CONFIG HEREE --------------------------------------- #
    # --------------------------------------------------- ↓↓↓ ---------------------------------------------------- #
    # ------------------------------------------------------------------------------------------------------------ #
    # RUN_CONFIG = "MNIST_CLASSIFIER_V1"
    # RUN_CONFIG = "CIFAR10_CLASSIFIER_V1"
    RUN_CONFIG = "CIFAR10_CLASSIFIER_V2"
    # ------------------------------------------------------------------------------------------------------------ #
    # --------------------------------------------------- ↑↑↑ ---------------------------------------------------- #
    # --------------------------------------- CHOOSE YOUR RUN CONFIG HEREE --------------------------------------- #

    project_conf = load_config("project_config.yaml")
    run_conf = load_config("run_configs.yaml")[RUN_CONFIG]

    train(project_config=project_conf, run_config=run_conf)
