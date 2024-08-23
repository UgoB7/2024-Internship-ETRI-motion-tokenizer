""" create data samples """
import pickle
import random
from collections import defaultdict

import lmdb
import math
import numpy as np
import os


class DataPreprocessor:
    def __init__(self, dataset_name, clip_lmdb_dir, out_lmdb_dir, n_poses, motion_fps):
        self.dataset_name = dataset_name
        self.n_poses = n_poses * 2  # 2x to support random shift augmentation
        self.subdivision_stride = n_poses
        self.motion_fps = motion_fps
        self.skip_stats = defaultdict(int)

        self.src_lmdb_env = lmdb.open(clip_lmdb_dir, readonly=True, lock=False)
        with self.src_lmdb_env.begin() as txn:
            self.n_videos = txn.stat()['entries']
        print(f"Initialized DataPreprocessor with {self.n_videos} videos")

        # delete existing lmdb cache
        try:
            os.remove(os.path.join(out_lmdb_dir, 'data.mdb'))
            os.remove(os.path.join(out_lmdb_dir, 'lock.mdb'))
        except OSError as e:
            pass

        # create db for samples
        max_map_size = int(1e11)  # 100 GB
        if "train" in out_lmdb_dir.lower():
            max_map_size = int(8e11)  # 800 GB 
        self.dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size=max_map_size)

    def run(self):
        src_txn = self.src_lmdb_env.begin(write=False)

        # sampling and normalization
        cursor = src_txn.cursor()
        for key, value in cursor:
            video = pickle.loads(value)
            vid = video['video_name']
            clips = video['clips']
            print(f"Processing video: {vid} with {len(clips)} clips")
            for clip_idx, clip in enumerate(clips):
                self._sample_from_clip(vid, clip)

        # print stats
        with self.dst_lmdb_env.begin() as txn:
            #print('no. of samples: ', txn.stat()['entries'])
            _=0

        # close db
        self.src_lmdb_env.close()
        self.dst_lmdb_env.sync()
        self.dst_lmdb_env.close()

        # print skip stats
        #print(self.skip_stats)

    def check_static_motion(self, skeletons, verbose=False):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, 3*joint_idx:3*(joint_idx+1)]
            variance = np.sum(np.var(wrist_pos.astype(np.float64), axis=0))
            assert not np.isinf(variance).any()
            return variance

        def get_motion_distance(skeleton, joint_idx):
            wrist_pos = skeleton[:, 3*joint_idx:3*(joint_idx+1)]
            dist = np.sum(np.abs(wrist_pos[1:] - wrist_pos[:-1]))
            return dist

        left_arm_dist = get_motion_distance(skeletons, 12)
        right_arm_dist = get_motion_distance(skeletons, 8)

        th = 120  # about 20-30%  TODO: avoid hard-coded value
        if left_arm_dist < th and right_arm_dist < th:
            if verbose:
                print('skip - check_static_motion left dist {}, right dist {}'.format(left_arm_dist, right_arm_dist))
            return True, max(left_arm_dist, right_arm_dist)
        else:
            if verbose:
                print('pass - check_static_motion left dist {}, right dist {}'.format(left_arm_dist, right_arm_dist))
            return False, max(left_arm_dist, right_arm_dist)

    def _sample_from_clip(self, vid, clip):
        clip_skeleton = clip['poses']
        clip_audio_raw = clip['audio_raw']

        if len(clip_skeleton) == 0 or len(clip_audio_raw) == 0:
            print(f'[skip] empty data {vid}')
            return

        if self.dataset_name == 'beat':
            clip_emotion = clip['emotion']
            clip_emotion_label = int(clip_emotion[0].split('_')[0])

            # select joints
            root_pos, joint_pos, joint_rot = clip_skeleton
            joint_rot = joint_rot[:].reshape(joint_rot.shape[0], -1)  # select joints and reshape
            joint_pos = joint_pos[:].reshape(joint_pos.shape[0], -1)

            # make motion vector
            clip_skeleton = np.concatenate([root_pos, joint_pos, joint_rot], axis=1)

        elif self.dataset_name == 'aihub':
            clip_emotion_label = -1  # none

            if type(clip_skeleton) == list and len(clip_skeleton) == 3:
                # bvh data
                root_pos, joint_pos, joint_rot = clip_skeleton
                joint_rot = joint_rot[:].reshape(joint_rot.shape[0], -1)
                joint_pos = joint_pos[:].reshape(joint_pos.shape[0], -1)

                clip_skeleton = np.concatenate([root_pos, joint_pos, joint_rot], axis=1)
            else:
                # 3d joint positions

                print("###############################################  3d joint positions")
                clip_skeleton = clip_skeleton[:].reshape(clip_skeleton.shape[0], -1)

            # downsample to 15 fps
            #clip_skeleton = clip_skeleton[::2]

        else:
            assert False, 'invalid dataset name'

        #print(f"Clip {vid} has {len(clip_skeleton)} poses after processing")

        # divide
        aux_info_list = []
        sample_skeletons_list = []
        sample_audio_list = []
        if 'transcription' in clip.keys():
            transcription = clip['transcription']
        else:
            transcription = None

        num_subdivision = math.floor(
            (len(clip_skeleton) - self.n_poses)
            / self.subdivision_stride) + 1  # floor((K - (N+M)) / S) + 1

        #print(f"Clip {vid} will be divided into {num_subdivision} subdivisions")

        for i in range(num_subdivision):
            start_idx = i * self.subdivision_stride
            end_idx = start_idx + self.n_poses
            start_time = start_idx / self.motion_fps
            end_time = end_idx / self.motion_fps

            sample_skeletons = clip_skeleton[start_idx:end_idx]

            static_motion, motion_var = self.check_static_motion(sample_skeletons, verbose=False)

            # data filtering

            skip_this_sample = False
            if 'start_end_frame' in clip.keys() and start_idx < clip['start_end_frame'][0]:
                skip_this_sample = True
                self.skip_stats['start_frame'] += 1
                print(f"[skip] start frame for {vid} subdivision {i}")
            elif 'start_end_frame' in clip.keys() and end_idx > clip['start_end_frame'][1]:
                skip_this_sample = True
                self.skip_stats['end_frame'] += 1
                print(f"[skip] end frame for {vid} subdivision {i}")
            elif static_motion:
                if random.random() < 0.95:  # allow some static samples (5%) for data diversity
                    skip_this_sample = True
                    #print(f"[skip] static motion for {vid} subdivision {i}")
                    self.skip_stats['motion'] += 1
            elif transcription:
                has_utterance = False
                for sentence in transcription:  # sorted by start_time of sentence
                    if sentence['start_time'] > end_time:
                        break
                    if max(sentence['start_time'], start_time) < min(sentence['end_time'], end_time):  # overlap
                        has_utterance = True
                if not has_utterance:
                    skip_this_sample = True
                    print(f"[skip] no utterance for {vid} subdivision {i}")
                    self.skip_stats['utt'] += 1

            if skip_this_sample:
                continue

            # raw audio
            audio_start = math.floor(start_idx / self.motion_fps * 16000)
            audio_end = audio_start + int(self.n_poses / self.motion_fps * 16000)
            if audio_end > len(clip_audio_raw):
                # print(f"[skip] audio end exceeds clip length for {vid} subdivision {i}")
                audio_end = len(clip_audio_raw)
            sample_audio = clip_audio_raw[audio_start:audio_end]

            transcription = 0 if transcription is None else transcription
            aux_info = {'vid': vid,
                        'start_frame_no': start_idx,
                        'end_frame_no': end_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'emotion': motion_var,  # todo: add motion space
                        'transcription': transcription  # todo: store sentences of interest, not entire transcription
                        }

            sample_skeletons_list.append(sample_skeletons)
            sample_audio_list.append(sample_audio)
            aux_info_list.append(aux_info)

        # save
        if len(sample_skeletons_list) > 0:
            #print(f"Saving {len(sample_skeletons_list)} samples from video {vid}")  # Debugging output
            with self.dst_lmdb_env.begin(write=True) as txn:
                for poses, audio, aux_info in zip(sample_skeletons_list, sample_audio_list, aux_info_list):
                    poses = np.asarray(poses)
                    k = '{:010}'.format(txn.stat()['entries']).encode('ascii')
                    txn.put(k, pickle.dumps([poses, audio, aux_info]), overwrite=False)
        else:
            print(f"No samples to save for video {vid}")
