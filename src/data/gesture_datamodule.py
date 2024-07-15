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
        dataset_name,
        motion_type,
        train_dir,
        val_dir,
        test_dir,
        batch_size,
        num_workers,
        pin_memory,
        n_poses,
        n_preposes,
        n_joints,
        pose_dim,
        motion_fps,
        raw_data_path,
        data_norm_stat_path,
        normalization_method,
        use_weighted_sampler
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
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
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_test:
            p = self.hparams

            data_stat = None
            if p.data_norm_stat_path:
                data_stat = np.load(p.data_norm_stat_path)

            if p.dataset_name == 'beat':
                self.data_train = BeatDataset(p.train_dir, p.n_poses, p.motion_fps, data_stat,
                                              p.normalization_method, random_shift=True)
                self.data_val = BeatDataset(p.val_dir, p.n_poses, p.motion_fps, data_stat,
                                            p.normalization_method, random_shift=False)
                self.data_test = BeatDataset(p.test_dir, p.n_poses, p.motion_fps, data_stat,
                                             p.normalization_method, random_shift=False)
            elif p.dataset_name == 'aihub':
                self.data_train = AihubDataset(p.train_dir, p.n_poses, p.motion_fps, data_stat,
                                               p.normalization_method, random_shift=True)
                self.data_val = AihubDataset(p.val_dir, p.n_poses, p.motion_fps, data_stat,
                                             p.normalization_method, random_shift=False)

            # weighted sampler
            if p.use_weighted_sampler:
                base_data_path = p.data_norm_stat_path.split(os.path.sep)[0]
                kmeans_results = pickle.load(open(os.path.join(base_data_path, 'kmeans.pkl'), 'rb'))
                stat_dict = kmeans_results['cluster_stat']
                labels = kmeans_results['labels']
                weights = [10000 / stat_dict[labels[i]] for i in range(len(self.data_train))]
                self.sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    def train_dataloader(self):
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

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # num_workers=0, # for debugging
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            # num_workers=0, # for debugging
            pin_memory=self.hparams.pin_memory,
            shuffle=False
        )
