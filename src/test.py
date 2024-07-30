from typing import Any, Dict, Optional, Tuple, List
import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import os
import subprocess
import torch
import wandb
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.utils.viz_util import unnormalize, render_aihub_motion
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

@task_wrapper
def test(cfg: DictConfig) -> Dict[str, Any]:
    """Tests the model using a specified checkpoint.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A dictionary with test metrics.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

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

    log.info("Starting testing!")
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path is None or ckpt_path == "":
        raise ValueError("Checkpoint path must be specified for testing!")
    
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    log.info(f"Tested using ckpt path: {ckpt_path}")

    # Generate and log videos
    log.info("Generating and logging videos...")
    generate_and_log_videos(model, datamodule, trainer)

    test_metrics = trainer.callback_metrics

    return test_metrics, None

def generate_and_log_videos(model: LightningModule, datamodule: LightningDataModule, trainer: Trainer):
    """Generates and logs videos for a few samples."""
    output_dir = trainer.default_root_dir
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    #print(f"######################## Test loader length: {len(test_loader)}")
    
    for batch_idx, batch in enumerate(test_loader):
        #print(f"############################## Batch {batch_idx}")
        if batch_idx == 0:

            features, _, _ = batch
            x_out, loss_commit, perplexity, codebook, code_idx = model(features)


            # Adjust sample_indices based on the actual number of samples in the batch
            sample_indices = [i for i in [0, 7, 15] if i < features.size(0)]

            for sample_idx in sample_indices:
                title = f"Sample {sample_idx}, Batch {batch_idx}"

                mp4_paths = []
                for target in ['target', 'generated']:
                    if target == 'target':
                        poses_npy = features[sample_idx].cpu().numpy()
                    elif target == 'generated':
                        poses_npy = x_out[sample_idx].detach().cpu().numpy()  # Detach tensor before converting to numpy

                    poses_npy = unnormalize(poses_npy, model.norm_method, model.data_stat)
                    n_joints = int(poses_npy.shape[1] / 3)
                    joint_pos = poses_npy[:, :n_joints * 3]

                    out_name = f'sample_{sample_idx}_batch_{batch_idx}_{target}.mp4'
                    out_mp4_path = render_aihub_motion(joint_pos, None, title, out_path=output_dir, out_name=out_name)
                    mp4_paths.append(out_mp4_path)

                # Generate token IDs string
                sample_code_idx = code_idx[sample_idx * 16:(sample_idx + 1) * 16].detach().cpu().numpy()
                token_ids_str = ', '.join(map(str, sample_code_idx))

                if len(mp4_paths) == 2:
                    out_path = os.path.join(output_dir, f'sample_{sample_idx}_batch_{batch_idx}_recon.mp4')
                    cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', mp4_paths[0], '-i', mp4_paths[1], '-filter_complex', 'hstack', out_path]
                    subprocess.call(cmd)
                    if trainer.logger is not None:
                        # Log the video and include token IDs in the metadata
                        video_log = wandb.Video(out_path, fps=30, format="mp4", caption=f"Tokens: {token_ids_str}")
                        wandb.log({
                            f"test/video_sample_{sample_idx}_batch_{batch_idx}": video_log
                        })

@hydra.main(version_base="1.3", config_path="../configs", config_name="test.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for testing.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with the test metric value.
    """
    extras(cfg)
    wandb.init(project=cfg.wandb_project, config=cfg)

    # test the model
    test_metrics, _ = test(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=test_metrics, metric_name=cfg.get("optimized_metric")
    )

    wandb.finish()

    # return optimized metric
    return metric_value

if __name__ == "__main__":
    main()
