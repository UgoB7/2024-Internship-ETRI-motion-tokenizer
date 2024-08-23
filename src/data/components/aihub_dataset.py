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

from src.utils.viz_util import render_aihub_motion, generate_bvh, render_bvh


class AihubDataset(Dataset):
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
        assert os.path.exists(preloaded_dir), f'run data_preprocess.py first (missing {preloaded_dir}'

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
        aux_info['transcription'] = 0  # todo: add transcription data

        # normalization
        if self.data_stat is not None:
            mean, std, data_max, data_min = self.data_stat[0], self.data_stat[1], self.data_stat[2], self.data_stat[3]
            if self.norm_method == 'min-max':
                max_min = np.clip(data_max - data_min, a_min=1e-5, a_max=None)
                pose_seq = (pose_seq - data_min) / max_min
            elif self.norm_method == 'mean-std':
                pose_seq = (pose_seq - mean) / std
            elif self.norm_method == 'none':
                pass
            else:
                assert False, 'unknown norm_method'

        # to tensors
        pose_seq = torch.from_numpy(pose_seq).reshape((pose_seq.shape[0], -1)).float()
        audio = torch.from_numpy(audio).float()

        if pose_seq.shape[0] == 0:
            raise ValueError(f"Empty pose sequence found at index {idx}")

        return pose_seq, audio, aux_info


def tst_aihub_dataset(base_path):  # test
    dataset = AihubDataset(
        os.path.join(base_path, 'lmdb_val'), n_poses=32, motion_fps=30,
        data_stat=np.load(os.path.join(base_path, "motion_data_stat.npy")),
        norm_method='none', random_shift=False
    )

    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=False)
    print('n_samples:', len(data_loader.dataset))

    show = True
    if show:
        for iter_idx, data in enumerate(data_loader, 0):
            poses, audio, aux_info = data

            if iter_idx == 0:
                print('audio shape:', audio.shape)
                print('pose shape:', poses.shape)

            for i in range(poses.size(0)):
                title = f"{aux_info['vid'][i]}, " + \
                        f"{str(datetime.timedelta(seconds=aux_info['start_time'][i].item()))}-" + \
                        f"{str(datetime.timedelta(seconds=aux_info['end_time'][i].item()))}"
                poses_npy = poses[i].cpu().numpy()
                print(aux_info['vid'][i])

                wav_path = os.path.join(base_path, f'temp/sample_{iter_idx}.wav')
                scipy.io.wavfile.write(wav_path, 16000, audio[i].cpu().numpy())

                if 'bvh' in base_path:
                    n_joints = int(poses_npy.shape[1] / 9)

                    # root_pos = poses_npy[:, :3]
                    # joint_pos = poses_npy[:, 3:n_joints*3]
                    # joint_rot = poses_npy[:, n_joints*3:].reshape(root_pos.shape[0], -1, 6)
                    # bvh_path = os.path.join(base_path, 'temp/sample.bvh')
                    # generate_bvh(root_pos, joint_rot, bvh_path, os.path.join(base_path, 'pymo_pipe.sav'))
                    # render_bvh(bvh_path, wav_path, title, os.path.join(base_path, 'temp'))

                    joint_pos = poses_npy[:, :n_joints*3]
                    render_aihub_motion(joint_pos, os.path.join(base_path, f'temp/sample_{iter_idx}.wav'), title,
                                        out_path=os.path.join(base_path, 'temp'))
                else:
                    render_aihub_motion(poses_npy, os.path.join(base_path, f'temp/sample_{iter_idx}.wav'), title,
                                        out_path=os.path.join(base_path, 'temp'))
                break

            if iter_idx > 10:
                break


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')

    lmdb_base_path = './data_aihub_bvh'
    start = time.time()
    tst_aihub_dataset(lmdb_base_path)
    print('elapsed:', time.time() - start)
