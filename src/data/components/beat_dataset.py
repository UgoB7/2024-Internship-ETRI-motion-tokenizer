import datetime
import logging
import math
import os
import pickle
import random
import time

import numpy as np
import lmdb as lmdb
import scipy.io.wavfile
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.utils.viz_util import generate_bvh, render_bvh


class BeatDataset(Dataset):
    def __init__(self, lmdb_dir, n_poses, motion_fps, data_stat, norm_method, random_shift):
        self.lmdb_dir = lmdb_dir
        self.n_poses = n_poses
        self.motion_fps = motion_fps
        self.data_stat = data_stat
        self.norm_method = norm_method
        self.random_shift = random_shift
        self.audio_len = int(self.n_poses / self.motion_fps * 16000)

        logging.info("Reading data '{}'...".format(lmdb_dir))
        preloaded_dir = lmdb_dir + f'_cache_{n_poses}'
        assert os.path.exists(preloaded_dir), 'run .py first'

        # init lmdb
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()['entries']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = '{:010}'.format(idx).encode('ascii')
            sample = txn.get(key)
            sample = pickle.loads(sample)
            pose_seq, audio, aux_info = sample

        # random shift
        if self.random_shift:
            shift = random.randint(0, pose_seq.shape[0] - self.n_poses)
        else:
            shift = 0
        pose_seq = pose_seq[shift:shift + self.n_poses]
        audio_shift = math.floor(shift / self.motion_fps * 16000)
        audio = audio[audio_shift:audio_shift + self.audio_len]
        aux_info['start_frame_no'] += shift
        aux_info['end_frame_no'] = aux_info['start_frame_no'] + self.n_poses
        aux_info['start_time'] = aux_info['start_frame_no'] / self.motion_fps
        aux_info['end_time'] = aux_info['end_frame_no'] / self.motion_fps

        # normalization
        if self.data_stat is not None:
            mean, std, data_max, data_min = self.data_stat[0], self.data_stat[1], self.data_stat[2], self.data_stat[3]
            if self.norm_method == 'min-max':
                max_min = np.clip(data_max - data_min, a_min=1e-5, a_max=None)
                pose_seq = (pose_seq - data_min) / max_min
            elif self.norm_method == 'mean-std':
                pose_seq = (pose_seq - mean) / std
            else:
                assert False, 'unknown norm_method'

        # to tensors
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(audio).float()

        return pose_seq, audio, aux_info


def tst_beat_dataset(base_path):  # test
    dataset = BeatDataset(
        os.path.join(base_path, 'lmdb_val'), n_poses=128, motion_fps=30,
        data_stat=np.load("data/motion_data_stat.npy"),
        norm_method='mean-std',
        random_shift=True
    )

    data_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=False)
    print('n_samples:', len(data_loader.dataset))
    vid_set = set()

    for i, batch in enumerate(tqdm(data_loader)):
        poses, audio, aux_info = batch

        if i == 0:
            print('audio shape:', audio.shape)
            print('pose shape:', poses.shape)
            print(aux_info)

        vid_set.update(aux_info['vid'])

        # break

    print(vid_set)

    show = False
    if show:
        for iter_idx, data in enumerate(data_loader, 0):
            poses, audio, aux_info = data

            for i in range(poses.size(0)):
                title = f"{aux_info['vid'][i]}, " + \
                        f"{str(datetime.timedelta(seconds=aux_info['start_time'][i].item()))}-" + \
                        f"{str(datetime.timedelta(seconds=aux_info['end_time'][i].item()))}"
                poses_npy = poses[i].cpu().numpy()
                print(aux_info['vid'][i])

                root_pos = poses_npy[:, :3]
                joint_rot = poses_npy[:, 3:].reshape(root_pos.shape[0], -1, 6)
                generate_bvh(root_pos, joint_rot, './data/temp/sample.bvh', 'data/pymo_pipe.sav')
                scipy.io.wavfile.write('./data/temp/sample.wav', 16000, audio.cpu().numpy())
                render_bvh('./data/temp/sample.bvh', './data/temp/sample.wav', title, out_path='./data/temp')
                # break


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')

    lmdb_base_path = './data'
    start = time.time()
    tst_beat_dataset(lmdb_base_path)
    print('elapsed:', time.time() - start)
