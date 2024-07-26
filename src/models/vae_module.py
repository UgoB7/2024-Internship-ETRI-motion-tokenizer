
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
        x = x.permute(0, 2, 1)
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the network.
        """
        x_in = self.preprocess(features)
        x_encoder = self.encoder(x_in)
        x_quantized, loss_commit, perplexity = self.quantizer(x_encoder)
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)

        return x_out, loss_commit, perplexity
    
    
    def compute_loss(self, pred_motion: Tensor, gt_motion: Tensor, loss_commit: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the overall loss and individual components.
        """
        loss_motion = self.reconstruction_loss(pred_motion, gt_motion)
        loss_vel = self.reconstruction_loss.forward_vel(pred_motion, gt_motion)
        total_loss = loss_motion + self.commit * loss_commit + self.loss_vel_factor * loss_vel
        return total_loss, loss_motion, loss_commit, loss_vel


    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Training step.
        """
        features, _, __ = batch
        x_out, loss_commit, perplexity = self(features)
        total_loss, loss_motion, loss_commit, loss_vel = self.compute_loss(x_out, features, loss_commit)
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
        x_out, loss_commit, perplexity = self(features)
        total_loss, loss_motion, loss_commit, loss_vel = self.compute_loss(x_out, features, loss_commit)
        batch_size = features.size(0)
        self.log('val_loss', total_loss, batch_size=batch_size)
        self.log('val_loss_motion', loss_motion, batch_size=batch_size)
        self.log('val_loss_commit', self.commit * loss_commit, batch_size=batch_size)
        self.log('val_loss_vel', self.loss_vel_factor * loss_vel, batch_size=batch_size)
        self.log('val_perplexity', perplexity)

        if batch_idx < 3:
            self._save_sample_video(features, x_out, batch_idx)

        return total_loss

    

    def _save_sample_video(self, features: Tensor, x_out: Tensor, batch_idx: int) -> None:
        """
        Save sample videos during validation.
        """
        output_dir = self.trainer.default_root_dir
        epoch = self.current_epoch
        sample_indices = [0, 7, 15]  # Indices of the samples you want to save videos for

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

            if len(mp4_paths) == 2:
                out_path = os.path.join(output_dir, f'sample_{sample_idx}_epoch_{epoch}_batch_{batch_idx}_recon.mp4')
                cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', mp4_paths[0], '-i', mp4_paths[1], '-filter_complex', 'hstack', out_path]
                subprocess.call(cmd)
                if self.trainer.logger is not None:
                    wandb.log({f"val/video_sample_{sample_idx}": wandb.Video(out_path, fps=30, format="mp4")})


    def validation_step_end(self, outputs):
        # Optional hook
        pass

    def on_validation_epoch_end(self):
        # Optional hook
        pass

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        Test step.
        """
        features, _, __ = batch
        x_out, loss_commit, perplexity = self(features)
        total_loss, loss_motion, loss_commit, loss_vel = self.compute_loss(x_out, features, loss_commit)
        batch_size = features.size(0)
        self.log('test_loss', total_loss, batch_size=batch_size)
        self.log('test_loss_motion', loss_motion, batch_size=batch_size)
        self.log('test_loss_commit', self.commit * loss_commit, batch_size=batch_size)
        self.log('test_loss_vel', self.loss_vel_factor * loss_vel, batch_size=batch_size)
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

    def encode(
        self,
        features: Tensor,
    ) -> Union[Tensor, Distribution]:

        N, T, _ = features.shape
        x_in = self.preprocess(features)
        

        x_encoder = self.encoder(x_in)
        
        x_encoder = self.postprocess(x_encoder)
        x_encoder = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder)
        code_idx = code_idx.view(N, -1)

        # latent, dist
        return code_idx, None
    

    def visualize_latent_space(self, data_loader):
        all_latents = []

        # Gather all latent vectors
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                features, _, __ = batch  # Unpack the batch and drop audio and aux_info
                # print(f"########################## Features shape: {features.shape} ##########################") # torch.Size([256, 128, 657])
                
                x_in = self.preprocess(features)
                # print(f"########################## x_in shape after preprocess: {x_in.shape} ##########################") # torch.Size([256, 657, 128])
                
                latents = self.encode(x_in)
                print(f"########################## Latents shape: {latents.shape} ##########################")
                
                all_latents.append(latents.cpu().numpy())
        
        all_latents = np.concatenate(all_latents, axis=0)
        print(f"########################## All latents concatenated shape: {all_latents.shape} ##########################")

        # Reduce dimensionality using t-SNE
        tsne = TSNE(n_components=2, verbose=1, random_state=123)
        tsne_results = tsne.fit_transform(all_latents)
        print(f"########################## t-SNE results shape: {tsne_results.shape} ##########################")

        # Plotting
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
        plt.title('Latent Space Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.show()



    def decode(self, z: Tensor):

        x_d = self.quantizer.dequantize(z)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()

        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out


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
