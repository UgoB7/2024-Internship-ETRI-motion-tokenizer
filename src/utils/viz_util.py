import sys
import os
import time

import numpy as np
import pygame
import torch
import joblib
import subprocess
from pytorch3d import transforms
from sklearn.pipeline import Pipeline
from textwrap import wrap

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

sys.path.append('./library')
from pymo.preprocessing import Numpyfier, MocapParameterizer
from pymo.parsers import BVHParser
from pymo.writers import BVHWriter

import logging
logger = logging.getLogger('matplotlib.animation')
logger.setLevel(logging.WARNING)

def unnormalize(poses, norm_method, data_stat):
    if norm_method == 'min-max':
        data_max, data_min = data_stat[2], data_stat[3]
        poses = poses * (data_max - data_min) + data_min
    elif norm_method == 'mean-std':
        mean, std = data_stat[0], data_stat[1]
        poses = poses * std + mean
    else:
        assert False

    return poses


def normalize(poses, norm_method, data_stat):
    if norm_method == 'min-max':
        data_max, data_min = data_stat[2], data_stat[3]
        poses = (poses - data_min) / (data_max - data_min)
    elif norm_method == 'mean-std':
        mean, std = data_stat[0], data_stat[1]
        poses = (poses - mean) / std
    else:
        assert False

    return poses


def generate_bvh(root_pos, joint_rot, out_bvh_path, pipeline_path):
    bvh_data = np.zeros((joint_rot.shape[0], 3 + joint_rot.shape[1] * 3))
    bvh_data[:, :3] = root_pos

    # 6d representation to euler angles
    mat = transforms.rotation_6d_to_matrix(torch.Tensor(joint_rot))
    angles = transforms.matrix_to_euler_angles(mat, "XYZ")
    angles = np.degrees(angles)
    bvh_data[:, 3:] = angles.reshape(angles.shape[0], -1)

    # write
    writer = BVHWriter()
    pipeline = joblib.load(pipeline_path)

    bvh_data = pipeline.inverse_transform([bvh_data])

    with open(out_bvh_path, 'w') as f:
        writer.write(bvh_data[0], f)


def render_bvh(bvh_path, wav_path, title=None, out_path='./data/mp4', save_npy=False, fps=15):
    bvh_file = os.path.basename(bvh_path)
    mp4_path = os.path.join(out_path, bvh_file.replace('.bvh', '.mp4'))

    joint_links = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),  # spine-head
                   (4, 10), (10, 11), (11, 12), (12, 13),  # right arm
                   (4, 42), (42, 43), (43, 44), (44, 45),  # left arm
                   (0, 74), (74, 75), (75, 76), (76, 77), (77, 78),  # right leg
                   (0, 81), (81, 82), (82, 83), (83, 84), (84, 85),  # left leg
                   (13, 14), (14, 15), (15, 16), (16, 17),  # right hand middle
                   (13, 19), (19, 20), (20, 21), (21, 22), (22, 23),  # right hand ring
                   (19, 25), (25, 26), (26, 27), (27, 28), (28, 29),  # right hand pinky
                   (13, 31), (31, 32), (32, 33), (33, 34), (34, 35),  # right hand index
                   (31, 37), (37, 38), (38, 39), (39, 40),  # right hand thumb
                   (45, 46), (46, 47), (47, 48), (48, 49),  # left hand middle
                   (45, 51), (51, 52), (52, 53), (53, 54), (55, 56),  # left hand ring
                   (51, 57), (57, 58), (58, 59), (59, 60), (60, 61),  # left hand pinky
                   (45, 63), (63, 64), (64, 65), (65, 66), (66, 67),  # left hand index
                   (63, 69), (69, 70), (70, 71), (71, 72)  # left hand thumb
                   ]

    # bvh2npy
    p = BVHParser()
    data_all = [p.parse(bvh_path)]

    data_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    npy_data = out_data[0]

    if save_npy:
        np.save(mp4_path.replace('.mp4', '_from_bvh.npy'), npy_data)

    # visualize
    x = npy_data.reshape((npy_data.shape[0], -1, 3))
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()

    def animate(i):
        if title:
            fig.suptitle('\n'.join(wrap(title + " " + str(i), 50)), fontsize='medium')

        pose = x[i]
        ax.clear()

        for j, pair in enumerate(joint_links):
            linewidth = 3 if j < 26 else 1
            ax.plot([pose[pair[0], 0], pose[pair[1], 0]],
                    [pose[pair[0], 2], pose[pair[1], 2]],
                    [pose[pair[0], 1], pose[pair[1], 1]],
                    zdir='z', linewidth=linewidth)
        lim = 75
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(lim, -lim)
        ax.set_zlim3d(-lim, lim)
        ax.set_xlabel('dim 0')
        ax.set_ylabel('dim 2')
        ax.set_zlabel('dim 1')
        ax.margins(x=0)

    num_frames = len(x)
    # num_frames = 10

    ani = animation.FuncAnimation(fig, animate, interval=1, frames=num_frames, repeat=False)
    ani.save(mp4_path, fps=fps, dpi=150)
    del ani
    plt.close(fig)
    # plt.show()

    # merge audio and video
    out_path = mp4_path.replace('.mp4', '_with_audio.mp4')
    cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', mp4_path, '-i', wav_path, '-c:v', 'copy', '-c:a', 'aac',
           '-shortest', out_path]
    print(cmd)
    subprocess.call(cmd)

    # remove video file
    if os.path.exists(out_path):
        os.remove(mp4_path)

    return out_path


def render_aihub_motion(poses, wav_path, title=None, out_path='./data_aihub/mp4', out_name=None, fps=30, fig=None):
    if out_path is not None:
        if out_name is None:
            mp4_path = os.path.join(out_path, 'motion_video.mp4')
        else:
            mp4_path = os.path.join(out_path, out_name)
    else:
        mp4_path = None

    joint_links = [(0, 1), (1, 2), (2, 3), (3, 4),
                   (2, 5), (5, 6), (6, 7), (7, 8),
                   (2, 9), (9, 10), (10, 11), (11, 12),
                   (0, 13), (13, 14), (14, 15), (15, 16),
                   (0, 17), (17, 18), (18, 19), (19, 20),
                   (0, 22), (22, 23), (23, 24), (24, 25)]

    # Check the shape of poses and log it
    #print(f"  ##################################### Poses shape: {poses.shape}")
    num_joints = poses.shape[1] // 9 
    #print(f"  ##################################### Number of joints in poses: {num_joints}")

    x = poses[:, :26*3].reshape((poses.shape[0], -1, 3))  # (n, j, 3)

    if not fig:
        fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()

    def animate(i):
        if title:
            fig.suptitle('\n'.join(wrap(title + " " + str(i), 40)), fontsize='medium')

        pose = x[i]
        ax.clear()

        for j, pair in enumerate(joint_links):
            # Check if pair indices are within bounds
            if pair[0] >= num_joints or pair[1] >= num_joints:
                print(f"Skipping invalid joint pair: {pair}")
                continue
            linewidth = 3 if j < 26 else 1
            ax.plot([pose[pair[0], 0], pose[pair[1], 0]],
                    [pose[pair[0], 2], pose[pair[1], 2]],
                    [pose[pair[0], 1], pose[pair[1], 1]],
                    zdir='z', linewidth=linewidth)

        lim = 80
        ax.set_xlim3d(-lim, lim)
        ax.set_ylim3d(lim, -lim)
        ax.set_zlim3d(-lim, lim)

        ax.set_xlabel('dim 0')
        ax.set_ylabel('dim 2')
        ax.set_zlabel('dim 1')
        ax.margins(x=0)

    num_frames = len(x)

    ani = animation.FuncAnimation(fig, animate, interval=33.33, frames=num_frames, repeat=False)

    if mp4_path:
        try:
            Writer = animation.writers['ffmpeg']
        except KeyError:
            raise RuntimeError("ffmpeg writer is not available in matplotlib. Make sure ffmpeg is installed and accessible.")
        
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(mp4_path, writer=writer)
        del ani
        plt.close(fig)
        return mp4_path
    else:
        # show
        plt.show()
        return None



def play_aihub_motion_with_audio(poses, wav_path, title=None, fps=30, fig=None):
    mp4_path = None

    joint_links = [(0, 1), (1, 2), (2, 3), (3, 4),
                   (2, 5), (5, 6), (6, 7), (7, 8),
                   (2, 9), (9, 10), (10, 11), (11, 12),
                   (0, 13), (13, 14), (14, 15), (15, 16),
                   (0, 17), (17, 18), (18, 19), (19, 20)]

    x = poses[:, :21*3].reshape((poses.shape[0], -1, 3))  # (n, j, 3)
    if not fig:
        fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    fig.suptitle('\n'.join(wrap(title, 40)), fontsize='medium')

    lim = 70
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(lim, -lim)
    ax.set_zlim3d(-lim, lim)

    ax.set_xlabel('dim 0')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 1')
    ax.margins(x=0)

    lines = [ax.plot([], [], [], zdir='z', linewidth=4 if j < 26 else 1)[0] for j in range(len(joint_links))]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def animate(i):
        start_time = time.time()

        pose = x[i]

        for j, (line, pair) in enumerate(zip(lines, joint_links)):
            line.set_data([pose[pair[0], 0], pose[pair[1], 0]],
                          [pose[pair[0], 2], pose[pair[1], 2]])
            line.set_3d_properties([pose[pair[0], 1], pose[pair[1], 1]])

        # 정확한 fps 유지
        elapsed_time = time.time() - start_time
        # pause_time = max(1. / fps - elapsed_time, 0)
        pause_time = max(0.028 - elapsed_time, 0)
        plt.pause(pause_time)

        return lines

    num_frames = len(x)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=1, blit=True, repeat=False)

    # play audio
    pygame.mixer.init()
    pygame.mixer.music.load(wav_path)
    pygame.mixer.music.play()

    plt.show()
    return None

def play_beat_motion_with_audio(poses, wav_path, title=None, fps=30, fig=None):
    mp4_path = None

    joint_links = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),  # spine-head
        (4, 9), (9, 10), (10, 11), (11, 12),  # right arm
        (4, 13), (13, 14), (14, 15), (15, 16),  # left arm
        (0, 17), (17, 18), (18, 19), (19, 20), (20, 21),  # right leg
        (0, 22), (22, 23), (23, 24), (24, 25), (25, 26),  # left leg
        (16, 27), (27, 28), (28, 29), (29, 30),  # LeftHandThumb
        (16, 31), (31, 32), (32, 33), (33, 34), (34, 35),  # LeftHandIndex
        (16, 36), (36, 37), (37, 38), (38, 39), (39, 40),  # LeftHandPinky
        (16, 41), (41, 42), (42, 43), (43, 44), (44, 45),  # LeftHandRing
        (16, 46), (46, 47), (47, 48), (48, 49),  # LeftHandMiddle (4 joints)
        (12, 50), (50, 51), (51, 52), (52, 53),  # RightHandThumb
        (12, 54), (54, 55), (55, 56), (56, 57), (57, 58),  # RightHandIndex
        (12, 59), (59, 60), (60, 61), (61, 62), (62, 63),  # RightHandPinky
        (12, 64), (64, 65), (65, 66), (66, 67), (67, 68),  # RightHandRing
        (12, 69), (69, 70), (70, 71), (71, 72)  # RightHandMiddle (4 joints)
    ]

    x = poses[:, :73*3].reshape((poses.shape[0], -1, 3))  # (n, j, 3)
    if not fig:
        fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    fig.suptitle('\n'.join(wrap(title, 40)), fontsize='medium')

    lim = 70
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(lim, -lim)
    ax.set_zlim3d(-lim, lim)

    ax.set_xlabel('dim 0')
    ax.set_ylabel('dim 2')
    ax.set_zlabel('dim 1')
    ax.margins(x=0)

    lines = [ax.plot([], [], [], zdir='z', linewidth=4 if j <= 26 else 1)[0] for j in range(len(joint_links))]

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def animate(i):
        start_time = time.time()

        pose = x[i]

        for j, (line, pair) in enumerate(zip(lines, joint_links)):
            line.set_data([pose[pair[0], 0], pose[pair[1], 0]],
                          [pose[pair[0], 2], pose[pair[1], 2]])
            line.set_3d_properties([pose[pair[0], 1], pose[pair[1], 1]])

        # 정확한 fps 유지
        elapsed_time = time.time() - start_time
        # pause_time = max(1. / fps - elapsed_time, 0)
        pause_time = max(0.022 - elapsed_time, 0)
        plt.pause(pause_time)

        return lines

    num_frames = len(x)

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=1, blit=True, repeat=False)

    # play audio
    pygame.mixer.init()
    pygame.mixer.music.load(wav_path)
    pygame.mixer.music.play()

    plt.show()
    return None


def make_side_by_side_video(path1, path2, output_dir, name):
    out_path = os.path.join(output_dir, name)
    cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', path1, '-i', path2, '-filter_complex', 'hstack',
           out_path]
    subprocess.call(cmd)
    return out_path
