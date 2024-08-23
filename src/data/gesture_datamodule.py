import pickle
from typing import Any, Dict, Optional, Tuple

import os
import torch
import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.components.beat_dataset import BeatDataset
from src.data.components.aihub_dataset import AihubDataset


class MotionDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        motion_type: str,
        test_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        n_poses: int,
        n_preposes: int,
        n_joints: int,
        pose_dim: int,
        motion_fps: int,
        raw_data_path: str,
        data_norm_stat_path: Optional[str] = None,
        normalization_method: Optional[str] = None,
        use_weighted_sampler: bool = False,
        train_dir: Optional[str] = None,
        val_dir: Optional[str] = None,
    ):
        super().__init__()

        # this line allows access to init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.sampler = None


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        #print(f"###################################Train dir: {self.hparams.train_dir}, Val dir: {self.hparams.val_dir}, Test dir: {self.hparams.test_dir}")
        #print(f"###################################not self.data_train and not self.data_val and not self.data_test: {not self.data_train and not self.data_val and not self.data_test}")
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            p = self.hparams
        

            #print(f"+++++++++++++++++++++++++++++Loading data for dataset: {p.dataset_name}")
            #print(f"Train dir: {p.train_dir}, Val dir: {p.val_dir}, Test dir: {p.test_dir}")

            data_stat = None
            if p.data_norm_stat_path:
                data_stat = np.load(p.data_norm_stat_path)

            if p.dataset_name == 'beat':
                if p.train_dir:
                    self.data_train = BeatDataset(p.train_dir, p.n_poses, p.motion_fps, data_stat,
                                                  p.normalization_method, random_shift=True)
                if p.val_dir:
                    self.data_val = BeatDataset(p.val_dir, p.n_poses, p.motion_fps, data_stat,
                                                p.normalization_method, random_shift=False)
                self.data_test = BeatDataset(p.test_dir, p.n_poses, p.motion_fps, data_stat,
                                             p.normalization_method, random_shift=False)
            elif p.dataset_name == 'aihub':
                if p.train_dir:
                    self.data_train = AihubDataset(p.train_dir, p.n_poses, p.motion_fps, data_stat,
                                                   p.normalization_method, random_shift=True)
                if p.val_dir:
                    self.data_val = AihubDataset(p.val_dir, p.n_poses, p.motion_fps, data_stat,
                                                 p.normalization_method, random_shift=False)
                self.data_test = AihubDataset(p.test_dir, p.n_poses, p.motion_fps, data_stat,
                                              p.normalization_method, random_shift=False)
            
            if self.data_train:
                print(f"Training dataset loaded with {len(self.data_train)} samples.")
            if self.data_val:
                print(f"Validation dataset loaded with {len(self.data_val)} samples.")
            if self.data_test:
                print(f"Test dataset loaded with {len(self.data_test)} samples.")

            # weighted sampler
            if p.use_weighted_sampler and self.data_train:
                base_data_path = p.data_norm_stat_path.split(os.path.sep)[0]
                kmeans_results = pickle.load(open(os.path.join(base_data_path, 'kmeans.pkl'), 'rb'))
                stat_dict = kmeans_results['cluster_stat']
                labels = kmeans_results['labels']
                weights = [10000 / stat_dict[labels[i]] for i in range(len(self.data_train))]
                self.sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))


    def train_dataloader(self):
        if self.data_train:
            shuffle = (self.sampler is None)
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                # num_workers=0, # for debugging
                pin_memory=self.hparams.pin_memory,
                shuffle=shuffle,
                sampler=self.sampler,
                drop_last=True
            )
        else:
            raise ValueError("Training data is not available.")

    def val_dataloader(self):
        if self.data_val:
            return DataLoader(
                dataset=self.data_val,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                # num_workers=0, # for debugging
                pin_memory=self.hparams.pin_memory,
                shuffle=False
            )
        else:
            raise ValueError("Validation data is not available.")

    def test_dataloader(self):
        if self.data_test:
            return DataLoader(
                dataset=self.data_test,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.num_workers,
                # num_workers=0, # for debugging
                pin_memory=self.hparams.pin_memory,
                shuffle=False
            )
        else:
            raise ValueError("Test data is not available.")
