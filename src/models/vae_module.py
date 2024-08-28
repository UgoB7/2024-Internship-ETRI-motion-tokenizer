
import wandb
import subprocess
import os
import scipy
import datetime
import numpy as np
from typing import Any, Dict, Tuple, Union
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import lightning.pytorch as pl
from lightning import LightningModule
from typing import List, Optional, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions.distribution import Distribution
from .tools.resnet import Resnet1D
from .tools.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
from collections import OrderedDict
from src.utils.viz_util import unnormalize, render_aihub_motion
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss: str, nb_joints: int):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        self.nb_joints = nb_joints
        self.motion_dim = nb_joints*3 + nb_joints*6               # OLD CODE DIFFERENT (nb_joints - 1) * 12 + 4 + 3 + 4
        
    def forward(self, motion_pred: Tensor, motion_gt: Tensor) -> Tensor:
        return self.Loss(motion_pred[..., :self.motion_dim], motion_gt[..., :self.motion_dim])
    
    # def forward_vel(self, motion_pred: Tensor, motion_gt: Tensor) -> Tensor:
    #     return self.Loss(motion_pred[..., 4 : (self.nb_joints - 1) * 3 + 4], motion_gt[..., 4 : (self.nb_joints - 1) * 3 + 4])

    def forward_vel(self, motion_pred: Tensor, motion_gt: Tensor) -> Tensor:
        # Calculate velocities for motion_pred and motion_gt
        velocity_pred = motion_pred[..., 1:, :] - motion_pred[..., :-1, :]
        velocity_gt = motion_gt[..., 1:, :] - motion_gt[..., :-1, :]

        # Calculate the loss using all features
        loss = self.Loss(velocity_pred, velocity_gt)

        return loss



class VQVaeLitModel(LightningModule):
    
    def __init__(self,
                 nfeats,
                 quantizer,
                 code_num,
                 code_dim,
                 output_emb_width,
                 down_t,
                 stride_t,
                 width,
                 depth,
                 dilation_growth_rate,
                 norm,
                 activation,
                 normalization_method,
                 data_norm_stat_path,
                 recons_loss,
                 nb_joints,
                 commit,
                 loss_vel_factor,
                 optimizer_cfg: dict,
                 scheduler_cfg: dict = None,
                 **kwargs) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.commit = commit
        self.loss_vel_factor = loss_vel_factor

        # Reconstruction loss
        self.reconstruction_loss = ReConsLoss(recons_loss, nb_joints)
        
        
        # Load data normalization stat
        self.data_stat = None
        self.data_norm_stat_path = data_norm_stat_path
        if data_norm_stat_path:
            self.data_stat = np.load(data_norm_stat_path)
            self.norm_method = normalization_method


        self.encoder = Encoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        self.decoder = Decoder(nfeats,
                               output_emb_width,
                               down_t,
                               stride_t,
                               width,
                               depth,
                               dilation_growth_rate,
                               activation=activation,
                               norm=norm)

        if quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(code_num, code_dim, mu=0.99)
        elif quantizer == "orig":
            self.quantizer = Quantizer(code_num, code_dim, beta=1.0)
        elif quantizer == "ema":
            self.quantizer = QuantizeEMA(code_num, code_dim, mu=0.99)
        elif quantizer == "reset":
            self.quantizer = QuantizeReset(code_num, code_dim)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # Mapping des noms aux classes PyTorch correspondantes
        self.optimizers_dict = {
            'torch.optim.AdamW': torch.optim.AdamW
        }
        self.schedulers_dict = {
            'torch.optim.lr_scheduler.CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR
        }

    def configure_optimizers(self):
        optimizer_class = self.optimizers_dict[self.optimizer_cfg['target']]
        optimizer = optimizer_class(self.parameters(), **self.optimizer_cfg['params'])
        
        if self.scheduler_cfg:
            scheduler_class = self.schedulers_dict[self.scheduler_cfg['target']]
            scheduler = scheduler_class(optimizer, **self.scheduler_cfg['params'])
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return optimizer

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        # print(f"[DEBUG] Preprocess - Input shape: {x.shape}")
        x = x.permute(0, 2, 1)
        # print(f"[DEBUG] Preprocess - Output shape: {x.shape}")
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        # print(f"[DEBUG] Postprocess - Input shape: {x.shape}")
        x = x.permute(0, 2, 1)
        # print(f"[DEBUG] Postprocess - Output shape: {x.shape}")
        return x

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the network.
        """
        # print(f"[DEBUG] Forward - Input features shape: {features.shape}")
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        # print(f"[DEBUG] Encoder output shape: {x_encoder.shape}")
        
        x_quantized, loss_commit, perplexity, codebook, code_idx = self.quantizer(x_encoder)  # include codebook and code_idx
        # print(f"[DEBUG] Quantizer output shapes: x_quantized={x_quantized.shape}, loss_commit={loss_commit}, perplexity={perplexity}, codebook={codebook.shape}, code_idx={code_idx.shape}")

        x_decoder = self.decoder(x_quantized)
        # print(f"[DEBUG] Decoder output shape: {x_decoder.shape}")
        
        x_out = self.postprocess(x_decoder)
        # print(f"[DEBUG] Forward - Output features shape: {x_out.shape}")
        
        return x_out, loss_commit, perplexity, codebook, code_idx
    
    
    def compute_loss(self, pred_motion: Tensor, gt_motion: Tensor, loss_commit: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the overall loss and individual components.
        """
        # print(f"[DEBUG] Compute Loss - pred_motion shape: {pred_motion.shape}, gt_motion shape: {gt_motion.shape}")
        loss_motion = self.reconstruction_loss(pred_motion, gt_motion)
        # print(f"[DEBUG] Reconstruction loss: {loss_motion.item()}")
        
        loss_vel = self.reconstruction_loss.forward_vel(pred_motion, gt_motion)
        # print(f"[DEBUG] Velocity loss: {loss_vel.item()}")
        
        total_loss = loss_motion + self.commit * loss_commit + self.loss_vel_factor * loss_vel
        # print(f"[DEBUG] Total loss: {total_loss.item()}")

        return total_loss, loss_motion, loss_commit, loss_vel


    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step.
        """
        # print(f"[DEBUG] Training Step - Batch index: {batch_idx}")
        features, _, __ = batch
        # print(f"[DEBUG] Batch features shape: {features.shape}") 

        try:
            x_out, loss_commit, perplexity, codebook, code_idx = self(features)
            total_loss, loss_motion, loss_commit, loss_vel = self.compute_loss(x_out, features, loss_commit)
        except Exception as e:
            print(f"[ERROR] Exception during training step at batch {batch_idx}: {e}")
            raise

        batch_size = features.size(0)
        self.log('train_loss', total_loss, batch_size=batch_size)
        self.log('train_loss_motion', loss_motion, batch_size=batch_size)
        self.log('train_loss_commit', self.commit * loss_commit, batch_size=batch_size)
        self.log('train_loss_vel', self.loss_vel_factor * loss_vel, batch_size=batch_size)
        self.log('train_perplexity', perplexity)

        return total_loss

    def training_step_end(self, outputs):
        # Optional hook
        pass

    def on_train_epoch_end(self):
        # Optional hook
        pass


    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Validation step.
        """
        features, audio, aux_info = batch
        x_out, loss_commit, perplexity, codebook, code_idx = self(features)
        total_loss, loss_motion, loss_commit, loss_vel = self.compute_loss(x_out, features, loss_commit)
        batch_size = features.size(0)
        self.log('val_loss', total_loss, batch_size=batch_size)
        self.log('val_loss_motion', loss_motion, batch_size=batch_size)
        self.log('val_loss_commit', self.commit * loss_commit, batch_size=batch_size)
        self.log('val_loss_vel', self.loss_vel_factor * loss_vel, batch_size=batch_size)
        self.log('val_perplexity', perplexity)

        if batch_idx < 1:
            self._save_sample_video(features, x_out, batch_idx, code_idx)
            self._log_codebook_to_wandb(codebook, code_idx, batch_idx, sample_indices=[1,10,100,200,250])

        return total_loss

    def _save_sample_video(self, features: Tensor, x_out: Tensor, batch_idx: int, code_idx: Tensor) -> None:
        """
        Save sample videos during validation.
        """
        output_dir = self.trainer.default_root_dir
        epoch = self.current_epoch
        sample_indices = [1,10,100,200,250]

        for sample_idx in sample_indices:
            title = f"Sample {sample_idx}, Epoch {epoch}, Batch {batch_idx}"

            mp4_paths = []
            for target in ['target', 'generated']:
                if target == 'target':
                    poses_npy = features[sample_idx].cpu().numpy()
                elif target == 'generated':
                    poses_npy = x_out[sample_idx].cpu().numpy()

                poses_npy = unnormalize(poses_npy, self.norm_method, self.data_stat)
                n_joints = int(poses_npy.shape[1] / 3)
                joint_pos = poses_npy[:, :n_joints * 3]

                out_name = f'sample_{sample_idx}_epoch_{epoch}_batch_{batch_idx}_{target}.mp4'
                out_mp4_path = render_aihub_motion(joint_pos, None, title, out_path=output_dir, out_name=out_name)
                mp4_paths.append(out_mp4_path)

            # Generate token IDs string
            sample_code_idx = code_idx[sample_idx * 16:(sample_idx + 1) * 16].detach().cpu().numpy()
            token_ids_str = ', '.join(map(str, sample_code_idx))

            if len(mp4_paths) == 2:
                out_path = os.path.join(output_dir, f'sample_{sample_idx}_epoch_{epoch}_batch_{batch_idx}_recon.mp4')
                cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', mp4_paths[0], '-i', mp4_paths[1], '-filter_complex', 'hstack', out_path]
                subprocess.call(cmd)
                if self.trainer.logger is not None:
                    # Log the video and include token IDs in the metadata
                    video_log = wandb.Video(out_path, fps=30, format="mp4", caption=f"Tokens: {token_ids_str}")
                    wandb.log({
                        f"val/video_sample_{sample_idx}_batch_{batch_idx}": video_log
                    })

    def _log_codebook_to_wandb(self, codebook, code_idx, batch_idx, sample_indices):
        # Visualize codebook
        plt.figure(figsize=(10, 8))
        sns.heatmap(codebook.detach().cpu().numpy(), cmap='viridis')
        plt.title(f'Codebook Visualization - Batch {batch_idx}')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        wandb.log({f'codebook_visualization_batch_{batch_idx}': wandb.Image(img_array)})
        plt.close()
        
        # Visualize code indices
        for sample_idx in sample_indices:
            sample_code_idx = code_idx[sample_idx * 16:(sample_idx + 1) * 16]
            plt.figure(figsize=(10, 8))
            sns.histplot(sample_code_idx.detach().cpu().numpy(), bins=self.quantizer.nb_code)
            plt.title(f'Code Indices Histogram - Sample {sample_idx} - Batch {batch_idx}')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            wandb.log({f'code_indices_histogram_sample_{sample_idx}_batch_{batch_idx}': wandb.Image(img_array)})
            plt.close()

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Test step.
        """
        features, _, __ = batch
        x_out, loss_commit, perplexity, codebook, code_idx = self(features) 
        total_loss, loss_motion, loss_commit, loss_vel = self.compute_loss(x_out, features, loss_commit)
        batch_size = features.size(0)
        self.log('test_loss', total_loss, batch_size=batch_size)
        self.log('test_loss_motion', loss_motion, batch_size=batch_size)
        self.log('test_loss_commit', loss_commit, batch_size=batch_size)
        self.log('test_loss_vel', loss_vel, batch_size=batch_size)
        self.log('test_perplexity', perplexity)

        return total_loss


    def test_step_end(self, outputs):
        # Optional hook
        pass

    def on_test_epoch_end(self):
        # Optional hook
        pass

    def any_extra_hook(self):
        # Any extra hooks or methods can be added here
        pass


class Encoder(nn.Module):

    def __init__(self,
                 input_emb_width,
                 output_emb_width,
                 down_t,
                 stride_t,
                 width,
                 depth,
                 dilation_growth_rate,
                 activation,
                 norm):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         activation=activation,
                         norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self,
                 input_emb_width,
                 output_emb_width,
                 down_t,
                 stride_t,
                 width,
                 depth,
                 dilation_growth_rate,
                 activation,
                 norm):
        super().__init__()
        blocks = []

        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width,
                         depth,
                         dilation_growth_rate,
                         reverse_dilation=True,
                         activation=activation,
                         norm=norm), nn.Upsample(scale_factor=2,
                                                 mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1))
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
