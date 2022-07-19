import os
import glob

import numpy as np
import scipy.io
import math

from scipy.io import loadmat
from scipy import signal

import torch

np.seterr(invalid='ignore')

class AMIGOSDataset:
    def __init__(self,
                 physiological_path="./data/physiological/processed/",
                 face_path='./data/face/',
                 is_train=True):
        self.sr = 128 ## sampling_rate
        self.WinLength = int(0.5 * self.sr)
        self.step = int(0.025 * self.sr)
        self.channels = 17

        self.tr_data, self.tr_label, self.ts_data, self.ts_label = self.load_data(face_path, physiological_path)

        if is_train:
            self.data = self.tr_data
            self.label = self.tr_label

        else:
            self.data = self.ts_data
            self.label = self.ts_label

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)

    def load_data(self, face_path, physiological_path):
        ps_dir = glob.glob(physiological_path+"*/*.mat")
        face_dir = glob.glob(face_path+"*/*.mov")
        ## 여기에 face_dir Number 순서대로 정렬

        tr_data, tr_label, ts_data, ts_label = self.ps_load(ps_dir)

        return tr_data, tr_label, ts_data, ts_label

    def ps_load(self, ps_dir):
        tr_list, tr_self_list, ts_list, ts_self_list = [], [], [], []

        for ps_file in ps_dir:
            raw_data = scipy.io.loadmat(ps_file)
            raw_ps = raw_data['joined_data'].squeeze() ### (20 Video Trial) ## (20, time, 17)
            raw_ps = raw_ps[:16] ## short videos (16, time, 17)

            self_annotations = raw_data['labels_selfassessment'].squeeze() ## (20, 12)
            external_annotations = raw_data['labels_ext_annotation'].squeeze()

            self_annotations = self_annotations[:16]
            external_annotations = external_annotations[:16] # (16, )

            processed_ps = np.array([self.generate_segment(ps_trial) for ps_trial in raw_ps]) ## (16, number of segment, time, channel)
            processed_self_annotations = np.array([np.full(proc_ps.shape[0], self_annotations[i].squeeze()[0]) for i, proc_ps in enumerate(processed_ps)]) ## (16, number of segments)

            ## train - test split (16, 12, 4)
            tr_ps, ts_ps, tr_annotations, ts_annotations = self.train_test_split(processed_ps, processed_self_annotations)

            tr_list.append(tr_ps)
            tr_self_list.append(tr_annotations)
            ts_list.append(ts_ps)
            ts_self_list.append(ts_annotations)

        tr_data = np.concatenate(tr_list) ## (Partc * Video Trial (12) * number of segment, time, channel)
        tr_label = np.concatenate(tr_self_list)  ## (Partc * Video Trial (12) * number of segment)
        ts_data = np.concatenate(ts_list)  ## (Partc * Video Trial (4) * number of segment, time, channel)
        ts_label = np.concatenate(ts_self_list)  ## (Partc * Video Trial (4) * number of segment)

        return torch.tensor(tr_data, dtype=torch.float32), torch.tensor(tr_label, dtype=torch.float32), \
               torch.tensor(ts_data, dtype=torch.float32), torch.tensor(ts_label, dtype=torch.float32)

    def normalize(self, ps_channel):
        # ps_channel = np.nan_to_num(ps_channel)
        normalized_ps = (ps_channel - ps_channel.mean(axis=0)) / ps_channel.std(axis=0) ## (time, )
        normalized_ps = np.nan_to_num(normalized_ps)

        return normalized_ps

    def generate_segment(self, ps_trial, frame_length=4):
        ## normalize
        ps_norm = np.stack([self.normalize(ps_trial[:,channel]) for channel in range(ps_trial.shape[1])], axis=1) ## (time, channel)

        ## cutoff
        step_size = self.sr // 2
        window_size = self.sr*frame_length

        ps_segment = np.stack([ps_norm[x:x+window_size] for x in range(0, ps_norm.shape[0], step_size) if x+window_size <= ps_norm.shape[0]], axis=0)

        return ps_segment

    def train_test_split(self, processed_ps, processed_self_annotations):
        np.random.seed(10)

        mask = np.ones(processed_ps.size, dtype=bool)
        test_index = np.random.choice(processed_ps.shape[0], 4) ## 16개 중에 4개 Test
        mask[test_index] = False

        tr_ps, ts_ps, tr_annotations, ts_annotations = processed_ps[mask], processed_ps[~mask], processed_self_annotations[mask], processed_self_annotations[~mask]
        ## (16, segment, time, channel) --> (16*segment, time, channel)
        tr_ps, ts_ps, tr_annotations, ts_annotations = np.concatenate(tr_ps), np.concatenate(ts_ps), np.concatenate(tr_annotations), np.concatenate(ts_annotations)

        return tr_ps, ts_ps, tr_annotations, ts_annotations