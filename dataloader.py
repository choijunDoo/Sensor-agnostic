import os
import glob

import numpy as np
import scipy.io
import math
from scipy import fftpack
from scipy.io import loadmat
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

import torch

import utils

np.seterr(invalid='ignore')

class AMIGOSDataset:
    def __init__(self,
                 physiological_path="./data/physiological/processed/",
                 face_path='./data/face/',
                 is_train=True):

        self.is_train = is_train
        self.sr = 128 ## sampling_rate
        self.frame_length = 4 # 2s
        self.step_length = 4 # 2s

        self.WinLength = int(self.sr * self.frame_length)
        self.step = int(self.sr * self.step_length)

        self.channels = 17

        self.data, self.label = self.load_data(face_path, physiological_path)

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)

    def load_data(self, face_path, physiological_path):
        ps_dir = glob.glob(physiological_path+"*/*.mat")
        face_dir = glob.glob(face_path+"*/*.mov")
        ## 여기에 face_dir Number 순서대로 정렬

        data, label = self.ps_load(ps_dir)

        return data, label

    def ps_load(self, ps_dir):
        data_list, label_list = [], []

        for ps_file in ps_dir:
            raw_data = scipy.io.loadmat(ps_file)
            raw_ps = raw_data['joined_data'].squeeze() ### (20 Video Trial) ## (20, time, 17)
            raw_ps = raw_ps[:16] ## short videos (16, time, 17)

            self_annotations = raw_data['labels_selfassessment'].squeeze() ## (20, 12)
            external_annotations = raw_data['labels_ext_annotation'].squeeze()

            self_annotations = self_annotations[:16] # (16, )
            external_annotations = external_annotations[:16]

            raw_ps, self_annotations, external_annotations = self.train_test_split(ps_file, self.is_train, raw_ps, self_annotations, external_annotations)

            # ps_list, index_list, channel_list = [], [], []
            # for index, raw in enumerate(raw_ps):
            #     raw = raw.squeeze()
            #     for channel in range(raw.shape[1]):
            #         if np.all(np.isnan(raw[:,channel])):
            #             ps_list.append(ps_file)
            #             index_list.append(index)
            #             channel_list.append(channel)
            #
            # print(ps_list)
            # print(index_list)
            # print(channel_list)

            # omission = []
            #
            # for index, annotation in enumerate(external_annotations):
            #     if annotation.size == 0:
            #         omission.append(index)
            #
            # mask = np.ones(raw_ps.shape[0], dtype=bool)
            # mask[omission] = False
            #
            # raw_ps = raw_ps[mask]
            # external_annotations = external_annotations[mask]

            # if raw_ps.size == 0:
            #     continue

            # Segments
            processed_ps = [self.generate_segment(ps_trial) for ps_trial in raw_ps] ## (16, number of segment, time, channel)

            # No Segments
            # processed_ps = np.array([np.stack([self.welchs_method(utils.nan_to_interp(ps_trial[:,channel]))
            #                                    for channel in range(ps_trial.shape[1])], axis=1)
            #                          for ps_trial in raw_ps]) ## (16, time, channel)

            # Segments
            processed_self_annotations = [np.full(proc_ps.shape[0], self_annotations[i].squeeze()[0]) for i, proc_ps in enumerate(processed_ps)] ## (16, number of segments)

            # No Segments
            # processed_self_annotations = np.array([self_annotation.squeeze()[1] for self_annotation in self_annotations])

            # processed_ext_annotations = np.array([self.external_segment(ext_annotation, proc_ps) for ext_annotation, proc_ps in zip(external_annotations, processed_ps)]) ## (16, number of segments)

            data_list.extend(processed_ps)
            label_list.extend(processed_self_annotations)

        data = np.concatenate(data_list) ## (Partc * Video Trial (12) * number of segment, time, channel)
        label = np.concatenate(label_list)  ## (Partc * Video Trial (12) * number of segment)

        # tr_data = np.array(tr_list) ## (Partc * Video Trial (12) * number of segment, time, channel)
        # tr_label = np.array(tr_self_list)  ## (Partc * Video Trial (12) * number of segment)
        # ts_data = np.array(ts_list)  ## (Partc * Video Trial (4) * number of segment, time, channel)
        # ts_label = np.array(ts_self_list)  ## (Partc * Video Trial (4) * number of segment)

        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

        # return tr_data, tr_label, ts_data, ts_label

    def normalize(self, ps_channel):
        # normalized_ps = (ps_channel - ps_channel.mean(axis=0)) / ps_channel.std(axis=0) ## (time, )
        # normalized_ps = np.nan_to_num(normalized_ps)

        # ps_channel = np.nan_to_num(ps_channel)

        scaler = MinMaxScaler()
        normalized_ps = scaler.fit_transform(ps_channel.reshape(-1,1))
        normalized_ps = normalized_ps.reshape(-1)

        return normalized_ps

    def generate_segment(self, ps_trial):
        ## normalize
        ps_norm = np.stack([utils.nan_to_interp(ps_trial[:,channel]) for channel in range(ps_trial.shape[1])], axis=1) ## (time, channel)
        ps_segment = np.stack([ps_norm[x:x+self.WinLength] for x in range(0, ps_norm.shape[0], self.step) if x+self.WinLength <= ps_norm.shape[0]], axis=0)

        ## FFT
        # ps_segment = np.stack([ps_trial[x:x + self.WinLength] for x in range(0, ps_trial.shape[0], self.step) if x + self.WinLength <= ps_trial.shape[0]], axis=0) # (segments, time, 17)

        # ps_fft = []
        # for segment in ps_segment:
        #     ffts = []
        #     for channel in range(segment.shape[1]):
        #         ffts.append(self.normalize(self.fft(segment[:, channel])))
        #
        #     fft_segment = np.stack(ffts, axis=1)
        #     ps_fft.append(fft_segment)
        #
        # ps_fft = np.stack(ps_fft, axis=0)

        return ps_segment

    def external_segment(self, ext_annotation, proc_ps):
        segments = proc_ps.shape[0] ## number of segments
        ext_annotation = ext_annotation[1:] # time (-5s ~ 0s)

        ext_segment = np.array([ext_annotation[(segment*self.step_length) // 20][2] if (segment*self.step_length // 20) < ext_annotation.shape[0] else ext_annotation[ext_annotation.shape[0]-1][1]
                            for segment in range(segments)]) ## (segment, )

        return ext_segment

    def train_test_split(self, file_name, is_train, raw_ps, self_annotations, external_annotations):
        np.random.seed(10)

        elim_video = utils.nan_elimination(file_name)

        test_index = np.random.choice(raw_ps.shape[0], 4) ## 16개 중에 4개 Test

        if is_train:
            mask = np.ones(raw_ps.shape[0], dtype=bool)
            mask[test_index] = False
        else:
            mask = np.zeros(raw_ps.shape[0], dtype=bool)
            mask[test_index] = True

        mask[elim_video] = False

        ps, self_annotation, ext_annotation  = raw_ps[mask], self_annotations[mask], external_annotations[mask]
        ## (16, segment, time, channel) --> (16*segment, time, channel)
        # tr_ps, ts_ps, tr_annotations, ts_annotations = np.concatenate(tr_ps), np.concatenate(ts_ps), np.concatenate(tr_annotations), np.concatenate(ts_annotations)

        return ps, self_annotation, ext_annotation

    def fft(self, sig):
        fft = fftpack.fft(sig) / sig.size
        fft = 2 * np.abs(fft)[:len(fft)//2]

        return fft

    def welchs_method(self, sig):
        segment = int(2 * self.sr)
        myhann = signal.get_window('hann', segment)

        myparams = dict(fs=self.sr, nperseg=segment, window=myhann, noverlap=segment/2,
                        scaling='spectrum', return_onesided=True)

        freq, ps = signal.welch(x=sig, **myparams)  # units uV**2
        # ps = 2*ps

        return np.sqrt(ps)