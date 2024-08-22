import os
import sys
import pickle
import random

from tqdm import tqdm
import hydra
import lmdb
import numpy as np
from omegaconf import DictConfig
from lightning.pytorch import LightningDataModule
from sklearn.cluster import MiniBatchKMeans


# Add the project root to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
from src.data.components.data_preprocessor import DataPreprocessor


def preprocess(lmdb_dir, cfg, dataset_name):
    print(f"Reading data {lmdb_dir}...")
    preloaded_dir = lmdb_dir + f'_cache_{cfg.n_poses}'

    # Debugging: Print directories and paths
    print(f"Dataset Name: {dataset_name}")
    print(f"LMDB Directory: {lmdb_dir}")
    print(f"Preloaded Directory: {preloaded_dir}")
    
    data_sampler = DataPreprocessor(dataset_name, lmdb_dir, preloaded_dir, cfg.n_poses, cfg.motion_fps)
    data_sampler.run()

    return preloaded_dir


def clustering(lmdb_path):
    batch_size = 200
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=batch_size)

    lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with lmdb_env.begin() as txn:
        n_samples = txn.stat()['entries']

        # kmeans fit
        minibatch = []
        for i in tqdm(range(n_samples)):
            key = '{:010}'.format(i).encode('ascii')
            pose_seq, audio, aux_info = pickle.loads(txn.get(key))
            pose_seq = pose_seq.flatten()
            minibatch.append(pose_seq)

            if len(minibatch) >= batch_size:
                kmeans = kmeans.partial_fit(minibatch)
                minibatch = []

        if len(minibatch) > 0:
            kmeans = kmeans.partial_fit(minibatch)

        # get cluster idx for each sample
        labels = []
        minibatch = []
        for i in tqdm(range(n_samples)):
            key = '{:010}'.format(i).encode('ascii')
            pose_seq, audio, aux_info = pickle.loads(txn.get(key))
            pose_seq = pose_seq.flatten()
            minibatch.append(pose_seq)

            if len(minibatch) >= batch_size:
                labels.extend(kmeans.predict(minibatch))
                minibatch = []

        if len(minibatch) > 0:
            labels.extend(kmeans.predict(minibatch))

    print(kmeans.cluster_centers_)
    print(len(labels))
    unique, count = np.unique(labels, return_counts=True)
    cluster_count = dict(zip(unique, count))
    print(cluster_count)
    print(f'min count: {min(count)}, max count: {max(count)}')
    pickle.dump({'kmeans_obj': kmeans, 'labels': labels, 'cluster_stat': cluster_count}, open('data/kmeans.pkl', 'wb'))


def temp_inspect_data(lmdb_path):
    import matplotlib.pyplot as plt
    fig = plt.figure()

    if False:
        ax = fig.add_subplot(projection='3d')

        lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with lmdb_env.begin() as txn:
            n_samples = txn.stat()['entries']

            i = random.randint(0, n_samples)
            key = '{:010}'.format(i).encode('ascii')
            pose_seq, audio, aux_info = pickle.loads(txn.get(key))
            print(key)
            print(aux_info)
            pose = pose_seq[0, :75]
            pose = pose.reshape(-1, 3)
            print(pose[9, 0])

            ax.scatter(pose[:, 0], pose[:, 2], pose[:, 1])
            ax.set_xlabel('x')
            # ax.set_xlim(-100, 100)
            ax.set_ylabel('z')
            ax.set_ylim(20, -20)
            ax.set_zlabel('y')
            # ax.set_zlim(-30, 30)
            plt.show()

    if True:
        import seaborn as sns

        data = []
        lmdb_env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with lmdb_env.begin() as txn:
            n_samples = txn.stat()['entries']

            for i in tqdm(range(0, n_samples, 15)):
                key = '{:010}'.format(i).encode('ascii')
                pose_seq, audio, aux_info = pickle.loads(txn.get(key))
                data.append(pose_seq)

                if np.mean(pose_seq[:, 30]) > 20:
                    print('r shoulder x:', np.mean(pose_seq[:, 30]))
                    print(aux_info)

        data = np.vstack(data)
        n_points = data.shape[0]

        joint_idx = 10  # right shoulder
        plt.subplot(3, 1, 1)
        sns.histplot(data[:, joint_idx * 3 + 0])
        plt.ylim(0, 100)
        plt.subplot(3, 1, 2)
        sns.histplot(data[:, joint_idx * 3 + 1])
        plt.ylim(0, 100)
        plt.subplot(3, 1, 3)
        sns.histplot(data[:, joint_idx * 3 + 2])
        plt.ylim(0, 100)
        plt.show()


@hydra.main(version_base="1.2", config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    # dataset_name = cfg.data.dataset_name
    # if dataset_name == 'beat':
    #     train_cache = preprocess(cfg.data.train_dir, cfg.data, dataset_name)
    #     #print('##########################################', cfg.data.train_dir)
    #     val_cache = preprocess(cfg.data.val_dir, cfg.data, dataset_name)
    #     test_cache = preprocess(cfg.data.test_dir, cfg.data, dataset_name)

    #     # train_cache = 'data/lmdb_train_cache'
    #     # clustering(train_cache)
    #     # temp_inspect_data(train_cache)
    # elif dataset_name == 'aihub':
    #     train_cache = preprocess(cfg.data.train_dir, cfg.data, dataset_name)
    #     val_cache = preprocess(cfg.data.val_dir, cfg.data, dataset_name)
    # else:
    #     assert False
    dataset_name = cfg.data.dataset_name
    train_cache = preprocess(cfg.data.train_dir, cfg.data, dataset_name)
    #print('##########################################', cfg.data.train_dir)
    val_cache = preprocess(cfg.data.val_dir, cfg.data, dataset_name)
    test_cache = preprocess(cfg.data.test_dir, cfg.data, dataset_name)


if __name__ == "__main__":
    main()
