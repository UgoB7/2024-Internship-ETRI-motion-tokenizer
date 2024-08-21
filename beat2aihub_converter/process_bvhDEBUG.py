
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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def process_bvh(bvh_path, dump_pipeline=False):

    target_joints = ['Hips', 'Spine', 'Spine1', 'Neck', 'Neck1', 'Head',
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
        ('np', Numpyfier())
    ])

    data_pipe_pos = Pipeline([
        ('param', MocapParameterizer('position')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)[0]
    print('############## out_data',out_data.shape)
    if dump_pipeline:
        print('dumping pipeline')
        joblib.dump(data_pipe, 'data/pymo_pipe.sav')

    out_data_pos = data_pipe_pos.fit_transform(data_all)[0]

    # euler angles to 6D representation
    root_position = out_data[:, :3]
    print('############### root_position:', root_position)
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

    print(root_position.shape)
    print(joint_rotations_6d.shape)
    print(joint_positions.shape)

    return root_position, joint_rotations_6d, joint_positions


root_position, joint_rotations_6d, joint_positions = process_bvh(r"D:\motion-tokenizer\BEAT_dataset\beat_english_v0.2.1KR\MM_D_C_FF_BB_S525S526_001_01_translated.bvh", dump_pipeline=True)
print('root_position:', root_position)
print('joint_rotations_6d:', joint_rotations_6d)
print('joint_positions:', joint_positions)