

import pickle
import sys
import os
import glob
import shutil
import torch
import numpy as np
import joblib
import librosa
import lmdb
import csv
from joblib import Parallel, delayed
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.pipeline import Pipeline
from scipy.signal import savgol_filter
from pytorch3d import \
    transforms  # installed by running: pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
from scipy.spatial.transform import Rotation as Rot
from scipy.io.wavfile import write as write_wav

# Add the project root to the sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)
#  print('project_root:', project_root)
from src.utils.viz_util import generate_bvh

sys.path.append('./library')
from pymo.preprocessing import Numpyfier, DownSampler, RootNormalizer, JointSelector, MocapParameterizer
from pymo.parsers import BVHParser


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    # from https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def process_bvh(bvh_path, dump_pipeline=False):
    target_joints = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head', 'HeadEnd',
                     'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                     'RightUpLeg', 'RightLeg', 'RightFoot', 'RightForeFoot', 'RightToeBase',
                     'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftForeFoot', 'LeftToeBase']
    target_joints.extend(['LeftHandThumb1', 'LeftHandThumb2', 'LeftHandThumb3', 'LeftHandThumb4',
        'LeftHandIndex', 'LeftHandIndex1', 'LeftHandIndex2', 'LeftHandIndex3', 'LeftHandIndex4',
        'LeftHandPinky', 'LeftHandPinky1', 'LeftHandPinky2', 'LeftHandPinky3', 'LeftHandPinky4',
        'LeftHandRing', 'LeftHandRing1', 'LeftHandRing2', 'LeftHandRing3', 'LeftHandRing4',
        'LeftHandMiddle1', 'LeftHandMiddle2', 'LeftHandMiddle3', 'LeftHandMiddle4',
        'RightHandThumb1', 'RightHandThumb2', 'RightHandThumb3', 'RightHandThumb4',
        'RightHandIndex', 'RightHandIndex1', 'RightHandIndex2', 'RightHandIndex3', 'RightHandIndex4',
        'RightHandPinky', 'RightHandPinky1', 'RightHandPinky2', 'RightHandPinky3', 'RightHandPinky4',
        'RightHandRing', 'RightHandRing1', 'RightHandRing2', 'RightHandRing3', 'RightHandRing4',
        'RightHandMiddle1', 'RightHandMiddle2', 'RightHandMiddle3', 'RightHandMiddle4'
    ])

    p = BVHParser()
    data_all = list()
    data_all.append(p.parse(bvh_path))

    # check default facing direction
    if data_all[0].skeleton['RightShoulder']['offsets'][2] < 0:
        rotate_root = True
        return None, None, None
    else:
        rotate_root = False

    data_pipe = Pipeline([
        ('root', RootNormalizer()),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        ('np', Numpyfier())
    ])

    data_pipe_pos = Pipeline([
        ('root', RootNormalizer()),
        ('param', MocapParameterizer('position')),
        ('jtsel', JointSelector(target_joints, include_root=False)),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)[0]
    if dump_pipeline:
        joblib.dump(data_pipe, 'new_data/pymo_pipe.sav')

    out_data_pos = data_pipe_pos.fit_transform(data_all)[0]

    # euler angles to 6D representation
    root_position = out_data[:, :3]
    joint_rotations = out_data[:, 3:].reshape((out_data.shape[0], -1, 3))  # (frame, joint, 3)
    mat = transforms.euler_angles_to_matrix(torch.Tensor(np.radians(joint_rotations)), "XYZ")

    if rotate_root:
        # TODO: need to handle motion files in different T-poses. Needs retargeting.
        # REF: https://blender.stackexchange.com/questions/72120/create-pose-from-existing-armature/72190#72190
        # This doesn't work properly
        # rot_mat = Rot.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix()
        rot_mat = Rot.from_euler('z', 180, degrees=True).as_matrix()
        rot_mat = rot_mat.transpose()
        mat[:, :] = np.matmul(rot_mat, mat[:, :])

    rot_6d = transforms.matrix_to_rotation_6d(mat)
    joint_rotations_6d = rot_6d.numpy()

    # joint positions
    joint_positions = out_data_pos[:, 3:].reshape((out_data_pos.shape[0], -1, 3))

    # check joint positions
    # exclude motions having unexpected shoulder positions
    # TODO: handle these files, not excluding
    rshoulder = np.mean(joint_positions[:, 9], axis=0)
    lshoulder = np.mean(joint_positions[:, 13], axis=0)
    angle = np.degrees(angle_between(rshoulder - lshoulder, (-1, 0, 0)))
    if angle > 50 or angle < -50:
        print('unexpected shoulder angle', angle)
        print(bvh_path)
        return None, None, None

    # smoothing
    root_position = savgol_filter(root_position, 51, 9, axis=0)
    joint_rotations_6d = savgol_filter(joint_rotations_6d, 51, 9, axis=0)
    joint_positions = savgol_filter(joint_positions, 51, 9, axis=0)

    # down sampling
    root_position = root_position[3::4, :]  # 120 -> 30 fps
    joint_rotations_6d = joint_rotations_6d[3::4, :]  # 120 -> 30 fps
    joint_positions = joint_positions[3::4, :]  # 120 -> 30 fps

    print(root_position.shape)
    print(joint_rotations_6d.shape)
    print(joint_positions.shape)

    return root_position, joint_rotations_6d, joint_positions


def thread_work(bvh_file):
    name = os.path.split(bvh_file)[1][:-4]
    print('processing', name)

    # load skeletons
    root_pos, joint_rot, joint_pos = process_bvh(bvh_file, dump_pipeline=False)
    if root_pos is None or joint_rot is None:
        print('wrong t-pose', bvh_file)
        return False
    out_bvh_path = os.path.join('new_data/processed_bvh', name + '.bvh')
    generate_bvh(root_pos, joint_rot, out_bvh_path, 'new_data/pymo_pipe.sav')

    # save
    root_pos = np.asarray(root_pos, dtype=np.float16)  # (n, 3)
    joint_rot = np.asarray(joint_rot, dtype=np.float16)  # (n, joints, 6)
    joint_pos = np.asarray(joint_pos, dtype=np.float16)  # (n, joints-1, 3)

    pkl_path = os.path.join('new_data/pkl', name + '.pkl')
    with open(pkl_path, 'wb') as f:
        d = {'bvh_path': bvh_file, 'root_pos': root_pos, 'joint_rot': joint_rot, 'joint_pos': joint_pos}
        pickle.dump(d, f)

    return True


def bvh2pkl(base_path):
    # print(base_path)
    bvh_files = sorted(glob.glob(os.path.join(base_path, '**/*.bvh'), recursive=True))
    # bvh_files = sorted(glob.glob(os.path.join(base_path, '1/1_wayne_0_1_8.bvh'), recursive=True))
    # bvh_files = sorted(glob.glob(os.path.join(base_path, '2/2_scott_0_1_1.bvh'), recursive=True))
    # bvh_files = bvh_files[:4]  # for debugging
    # print(bvh_files)

    for bvh_file in bvh_files:  # dump pipeline
        if '1_wayne_0_118_118' in bvh_file:  # TODO: choose a better one (having a rest finger pose at the first frame)
            print(bvh_file)
            process_bvh(bvh_file, dump_pipeline=True)
            break

    delayed_funs = []

    print(len(bvh_files), 'bvh files')
    for i, bvh_file in enumerate(bvh_files):
        print(bvh_file)

        # add process
        delayed_funs.append(delayed(thread_work)(bvh_file))

    # run
    results = Parallel(n_jobs=10)(delayed_funs)
    # results = Parallel(n_jobs=1)(delayed_funs)


def make_lmdb():
    exclude_list = ['10_kieks_1_1_1', '10_kieks_1_2_2', '16_jorge_1_3_3', '16_jorge_5_3_3', '18_daiki_1_1_1', 
                    '25_goto_1_1_1', '25_goto_1_2_2', '25_goto_1_3_3']

    # delete existing lmdb
    try:
        os.remove('new_data/lmdb_train/data.mdb'), os.remove('new_data/lmdb_train/lock.mdb')
        os.remove('new_data/lmdb_val/data.mdb'), os.remove('new_data/lmdb_val/lock.mdb')
        os.remove('new_data/lmdb_test/data.mdb'), os.remove('new_data/lmdb_test/lock.mdb')
    except OSError as e:
        pass

    # create lmdb
    entry_idx = 0
    max_map_size = int(1e11)  # 100 GB
    db = [lmdb.open(os.path.join('new_data', 'lmdb_train'), map_size=max_map_size),
          lmdb.open(os.path.join('new_data', 'lmdb_val'), map_size=max_map_size),
          lmdb.open(os.path.join('new_data', 'lmdb_test'), map_size=max_map_size)]

    # load pkl
    all_poses = []
    pkl_files = sorted(list(set(glob.glob('new_data/pkl/*.pkl')) - set(glob.glob('new_data/pkl/*_mirror.pkl'))))

    print(len(pkl_files), 'pkl files')
    for i, pkl_file in enumerate(pkl_files):
        if os.path.splitext(os.path.basename(pkl_file))[0] in exclude_list:
            print('skip!', pkl_file)
            continue

        print(pkl_file)
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        root_pos = data['root_pos']
        joint_rot = data['joint_rot']
        joint_pos = data['joint_pos']

        # load audio
        wav_path = data['bvh_path'].replace('.bvh', '.wav')
        if os.path.isfile(wav_path) is False:
            print(f'cannot find {wav_path}')
            continue
        audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')

        # load aux info
        bvh_path = data['bvh_path']
        csv_path = bvh_path.replace('.bvh', '.csv')
        if os.path.isfile(csv_path) is False:
            emotion_tag = '00_netural'
        else:
            with open(csv_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                csv_data = list(csv_reader)
            assert len(csv_data) == 1
            emotion_tag = csv_data[0]

        # write to db
        if entry_idx % 20 == 1:
            dataset_idx = 2  # test
        elif entry_idx % 20 == 2:
            dataset_idx = 1  # test
        else:
            dataset_idx = 0  # train

        name = os.path.split(bvh_path)[1][:-4]
        data_all = np.concatenate((root_pos,
                                   joint_pos.reshape(joint_pos.shape[0], -1),
                                   joint_rot.reshape(joint_rot.shape[0], -1)), axis=1)

        with db[dataset_idx].begin(write=True) as txn:
            k = '{:010}'.format(txn.stat()['entries']).encode('ascii')
            v = pickle.dumps({'video_name': name,
                              'clips': [{'poses': [root_pos, joint_pos, joint_rot], 'audio_raw': audio_raw,
                                         'emotion': emotion_tag
                                         }]})
            txn.put(k, v, overwrite=False)

            # use mirrored motion for train set
            pkl_file_mirror = pkl_file.replace('.bvh', '_mirror.pkl')
            assert os.path.exists(pkl_file_mirror)
            if dataset_idx == 0 and os.path.exists(pkl_file_mirror):
                with open(pkl_file_mirror, 'rb') as f:
                    data = pickle.load(f)
                    root_pos = data['root_pos']
                    joint_rot = data['joint_rot']
                    joint_pos = data['joint_pos']

                k = '{:010}'.format(txn.stat()['entries']).encode('ascii')
                v = pickle.dumps({'video_name': name + '_mirror',
                                  'clips': [{'poses': [root_pos, joint_pos, joint_rot], 'audio_raw': audio_raw,
                                             'emotion': emotion_tag
                                             }]})
                txn.put(k, v, overwrite=False)

        all_poses.append(data_all)
        entry_idx += 1

    # count entries and close
    print('n_total_entries', entry_idx)
    for j in range(len(db)):
        with db[j].begin(write=True) as txn:
            keys = [key for key, _ in txn.cursor()]
            print('number of db entries: ', txn.stat()['entries'])
            print('last key: ', keys[-1])

        # close
        db[j].sync()
        db[j].close()

    # calculate pose data statistics
    print('calculating data stat...')
    all_poses = np.vstack(all_poses[::3])  # too many, use a part of them
    pose_mean = np.mean(all_poses, axis=0, dtype=np.float64)
    pose_std = np.std(all_poses, axis=0, dtype=np.float64)
    pose_std = np.clip(pose_std, a_min=0.01, a_max=None)  # clipping
    pose_max = np.max(all_poses, axis=0)
    pose_min = np.min(all_poses, axis=0)

    print('data mean, std')
    print(pose_mean.shape)
    print(pose_std.shape)
    print('data_mean:', str(["{:0.5f}".format(e) for e in pose_mean]).replace("'", ""))
    print('data_std:', str(["{:0.5f}".format(e) for e in pose_std]).replace("'", ""))
    np.save('new_data/motion_data_stat.npy', np.array(np.stack((pose_mean, pose_std, pose_max, pose_min))))


def find_first_bvh_file(base_path: str):
    """Find the first BVH file in the directory."""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".bvh"):
                return os.path.join(root, file)
    return None


def find_first_wav_file(base_path: str):
    """Find the first WAV file in the directory."""
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".wav"):
                return os.path.join(root, file)
    return None


def clone_file(file_path: str, num_clones: int = 11):
    """Clone a file a specified number of times in the same directory."""
    base_dir, original_file_name = os.path.split(file_path)
    file_name, file_extension = os.path.splitext(original_file_name)
    for i in range(1, num_clones + 1):
        new_file_name = f"{file_name}_clone_{i}{file_extension}"
        new_file_path = os.path.join(base_dir, new_file_name)
        shutil.copyfile(file_path, new_file_path)
        #print(f"Cloned {file_path} to {new_file_path}")


def remove_clones(base_path: str, file_extension: str):
    """Remove all cloned files with the specified extension in the directory."""
    clone_files = glob.glob(os.path.join(base_path, f"*clone*{file_extension}"))
    for file_path in clone_files:
        os.remove(file_path)
        #print(f"Removed clone {file_path}")


@hydra.main(version_base=None, config_path="configs", config_name="eval.yaml")
def main(cfg):
    base_path = cfg['paths']['data_dir_test']
    normalized_base_path_forWindowsUsers = os.path.normpath(base_path)
    # print('normalized_base_path_forWindowsUsers:', normalized_base_path_forWindowsUsers)
    base_path = normalized_base_path_forWindowsUsers
    
    # Find and clone the first BVH file
    first_bvh_file = find_first_bvh_file(base_path)
    if first_bvh_file:
        clone_file(first_bvh_file)
    else:
        print("No BVH file found in the specified directory.")
    
    # Find and clone the first WAV file
    first_wav_file = find_first_wav_file(base_path)
    if first_wav_file:
        clone_file(first_wav_file)
    else:
        print("No WAV file found in the specified directory.")
    
    bvh2pkl(base_path)
    make_lmdb()

    # Remove all cloned files
    remove_clones(base_path, ".bvh")
    remove_clones(base_path, ".wav")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
