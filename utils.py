import re
import os

import numpy as np

# https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

null_eeg = {'12': [4],
            '21': [1,10],
            '22':[15],
            '23':[0,4,6,8,11],
            '24': [0,7,11,12,13,14,15],
            '33': [0,1,2,6,7,8,9,10,12,15]
            }

null_ecg = {'09': [0,1,2,5,6,8,10,11,12,14,15]}

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def nan_to_interp(y):
    if np.all(np.isnan(y)):
       y = np.nan_to_num(y)
    else:
        nans, x = nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])
    return y

def nan_elimination(file_name):
    file_name = os.path.basename(file_name)
    partc_num = re.findall('\d+', file_name)[0]

    if partc_num in null_eeg.keys():
        elim_video = null_eeg.get(partc_num)
    elif partc_num in null_ecg.keys():
        elim_video = null_ecg.get(partc_num)
    else:
        elim_video = []

    return elim_video