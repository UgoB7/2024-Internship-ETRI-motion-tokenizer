import re
import pickle
import sys
import os
import glob
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

    target_joints = ['Hips', 'Spine', 'Spine1', 'Neck', 'Head',
                     'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                     'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                     'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
                     'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase']


    p = BVHParser()
    data_all = list()
    data_all.append(p.parse(bvh_path))

    # # check default facing direction
    # if data_all[0].skeleton['RightShoulder']['offsets'][2] < 0:
    #     print(data_all[0].skeleton['RightShoulder']['offsets'][2])
    #     print('wrong facing direction', bvh_path)
    #     rotate_root = True
    #     return None, None, None
    # else:
    #     rotate_root = False

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
    #print('############## out_data',out_data.shape)
    if dump_pipeline:
        print('dumping pipeline')
        joblib.dump(data_pipe, 'data/pymo_pipe.sav')

    #print('############## len(target_joints):', len(target_joints))
    #print('############## out_data.shape[1]:', out_data.shape[1])
    if  out_data.shape[1] != len(target_joints) * 3 + 3:
        #print('################### wrong joint number:', out_data.shape[1],  len(target_joints) * 3 + 3)
        return None, None, None

    out_data_pos = data_pipe_pos.fit_transform(data_all)[0]

    # euler angles to 6D representation
    root_position = out_data[:, :3]
    #print('############### root_position:', root_position)
    joint_rotations = out_data[:, 3:].reshape((out_data.shape[0], -1, 3))  # (frame, joint, 3)
    mat = transforms.euler_angles_to_matrix(torch.Tensor(np.radians(joint_rotations)), "XYZ")

    if rotate_root:
        print('rotating root')
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
    # rshoulder = np.mean(joint_positions[:, 6], axis=0)
    # lshoulder = np.mean(joint_positions[:, 10], axis=0)
    # angle = np.degrees(angle_between(rshoulder - lshoulder, (-1, 0, 0)))
    # if angle > 50 or angle < -50:
    #     print('unexpected shoulder angle', angle)
    #     print(bvh_path)
    #     return None, None, None

    # smoothing
    root_position = savgol_filter(root_position, 51, 9, axis=0)
    joint_rotations_6d = savgol_filter(joint_rotations_6d, 51, 9, axis=0)
    joint_positions = savgol_filter(joint_positions, 51, 9, axis=0)

    # # down sampling
    # root_position = root_position[3::4, :]  # 120 -> 30 fps
    # joint_rotations_6d = joint_rotations_6d[3::4, :]  # 120 -> 30 fps
    # joint_positions = joint_positions[3::4, :]  # 120 -> 30 fps

    #print(root_position.shape)
    #print(joint_rotations_6d.shape)
    #print(joint_positions.shape)

    return root_position, joint_rotations_6d, joint_positions


def thread_work(bvh_file):
    name = os.path.split(bvh_file)[1][:-4]
    print('processing:', bvh_file)

    # load skeletons
    root_pos, joint_rot, joint_pos = process_bvh(bvh_file, dump_pipeline=False)
    #print('root_pos: /n', root_pos.shape)
    #print('joint_rot:  /n', joint_rot.shape)
    if root_pos is None or joint_rot is None:
        print('wrong t-pose', bvh_file)
        return False
    out_bvh_path = os.path.join('data/processed_bvh', name + '.bvh')
    generate_bvh(root_pos, joint_rot, out_bvh_path, 'data/pymo_pipe.sav')

    # save
    root_pos = np.asarray(root_pos, dtype=np.float16)  # (n, 3)
    joint_rot = np.asarray(joint_rot, dtype=np.float16)  # (n, joints, 6)
    joint_pos = np.asarray(joint_pos, dtype=np.float16)  # (n, joints-1, 3)

    pkl_path = os.path.join('data/pkl', name + '.pkl')
    with open(pkl_path, 'wb') as f:
        d = {'bvh_path': bvh_file, 'root_pos': root_pos, 'joint_rot': joint_rot, 'joint_pos': joint_pos}
        pickle.dump(d, f)

    return True


def bvh2pkl(aihub_base_path, beat_base_path):
    # print(base_path)
    bvh_files_aihub = sorted(glob.glob(os.path.join(aihub_base_path, '**/*.bvh'), recursive=True))
    bvh_files_beat = sorted(glob.glob(os.path.join(beat_base_path, '**/*.bvh'), recursive=True))
    print(f"Number of BVH files in AIHub directory: {len(bvh_files_aihub)}")
    print(f"Number of BVH files in BEAT directory: {len(bvh_files_beat)}")
    # bvh_files = sorted(glob.glob(os.path.join(base_path, '1/1_wayne_0_1_8.bvh'), recursive=True))
    # bvh_files = sorted(glob.glob(os.path.join(base_path, '2/2_scott_0_1_1.bvh'), recursive=True))
    # bvh_files = bvh_files[:4]  # for debugging
    # print(bvh_files)

    bvh_files_aihub = [file for file in bvh_files_aihub if file.endswith('_translated.bvh')]
    bvh_files_beat = [file for file in bvh_files_beat if file.endswith('_translated.bvh')]
    print(f"Number of BVH files translated in AIHub directory: {len(bvh_files_aihub)}")
    print(f"Number of BVH files translated in BEAT directory: {len(bvh_files_beat)}")

    bvh_files = bvh_files_aihub + bvh_files_beat
    print(f"Total BVH files to process: {len(bvh_files)}")
    
    for bvh_file in bvh_files:  # dump pipeline
        if 'MM_D_E_FF_CC_S335S336_001_01_translated' in bvh_file:  # TODO: choose a better one (having a rest finger pose at the first frame)
            print(bvh_file)
            process_bvh(bvh_file, dump_pipeline=True)
            break

    delayed_funs = []

    #print(len(bvh_files), 'bvh files')
    for i, bvh_file in enumerate(bvh_files):
        #print(bvh_file)

        # add process
        delayed_funs.append(delayed(thread_work)(bvh_file))

    # run
    results = Parallel(n_jobs=-1)(delayed_funs)
    #results = Parallel(n_jobs=1)(delayed_funs)


def make_lmdb():
    exclude_list = ['10_kieks_1_1_1', '10_kieks_1_2_2', '16_jorge_1_3_3', '16_jorge_5_3_3', '18_daiki_1_1_1',    # moving around. not facing front
                    '25_goto_1_1_1', '25_goto_1_2_2', '25_goto_1_3_3']  # root position error

    # delete existing lmdb
    try:
        os.remove('data/lmdb_train/data.mdb'), os.remove('data/lmdb_train/lock.mdb')
        os.remove('data/lmdb_val/data.mdb'), os.remove('data/lmdb_val/lock.mdb')
        os.remove('data/lmdb_test/data.mdb'), os.remove('data/lmdb_test/lock.mdb')
    except OSError as e:
        pass

    # create lmdb
    entry_idx = 0
    max_map_size = int(1e11)  # 100 GB
    db = [lmdb.open(os.path.join('data', 'lmdb_train'), map_size=max_map_size),
          lmdb.open(os.path.join('data', 'lmdb_val'), map_size=max_map_size),
          lmdb.open(os.path.join('data', 'lmdb_test'), map_size=max_map_size)]

    # load pkl
    all_poses = []
    pkl_files = sorted(list(set(glob.glob('data/pkl/*.pkl')) - set(glob.glob('data/pkl/*_mirror.pkl'))))

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

        # # load audio
        # wav_path = data['bvh_path'].replace('.bvh', '.wav')
        # if os.path.isfile(wav_path) is False:
        #     print(f'cannot find {wav_path}')
        #     continue
        # audio_raw, audio_sr = librosa.load(wav_path, mono=True, sr=16000, res_type='kaiser_fast')

        audio_sr = 16000 
        duration_in_minutes = 2  
        duration_in_seconds = duration_in_minutes * 60  
        audio_raw = np.zeros(int(audio_sr * duration_in_seconds))

        # load aux info
        bvh_path = data['bvh_path']
        csv_path = bvh_path.replace('.bvh', '.csv')
        if os.path.isfile(csv_path) is False:
            # print(f'cannot find {csv_path}')
            # continue

            # use default emotion tag
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
    # all_poses = np.vstack(all_poses)
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
    np.save('data/motion_data_stat.npy', np.array(np.stack((pose_mean, pose_std, pose_max, pose_min))))


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg):
    aihub_base_path = cfg['paths']['aihub_data_dir']
    beat_base_path = cfg['paths']['beat_data_dir']
    aihub_base_path = os.path.normpath(aihub_base_path)
    beat_base_path = os.path.normpath(beat_base_path)

    bvh2pkl(aihub_base_path, beat_base_path)
    #make_lmdb()


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
