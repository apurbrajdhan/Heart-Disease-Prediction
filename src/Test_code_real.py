from filtering import *

import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from scipy.signal import freqz    # Computes the frequency response of a digital filter
#from keras.models import Sequential
#from keras.layers import Dense
import tensorflow as tf
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.models import Sequential,load_model
#from keras.models import 
#from tensorflow.python.keras.layers import Input, Dense



import tkinter.filedialog
     

xc=tkinter.filedialog.askopenfilename()


sampling_freq = 1000
cutoff_freq = 100

low_cutoff_freq = 200
high_cutoff_freq = 300





sampfrom = 0
sampto = 2000



rec_index = xc[len(xc)-7:len(xc)-4]
sampfrom = 0
sampto = 1000

import wfdb

wfdb.show_ann_labels()

import wfdb
'''
for rec in DS1:
    annotations = wfdb.rdann("mitdb/" + rec, "atr")
    annotations.get_contained_labels()
    #print(annotations.contained_labels)
    #print()
'''
sampfrom = 0
sampto = 2000

import cv2

# Load an color image in grayscale


cv2.waitKey(0)

#rec_index = "119"
sampfrom = 0
sampto = 1000


def autocorr (x, mode="full"):
    y = np.convolve(x, x, mode)
    return y[int(y.size/2) :] # / y[int(y.size/2)]    # Normalized



import wfdb
from statsmodels.tsa.stattools import levinson_durbin
from collections import OrderedDict

from filtering import butter_filter

def extract_features (record_path, length_qrs, length_stt, ar_order_qrs, ar_order_stt, sampfrom=0, sampto=-1, use_filter=True):

    """
    A list holding tuples with values 'N' or 'VEB', and the length in samples of each corresponding QRS
    and ST/T complexes, plus the length in samples of pre- and post-RR
    """
    print(record_path)
    qrs_stt_rr_list = list()
    sampto = 1000
    #print(sampto)

    if sampto < 0:
        raw_signal, _ = wfdb.rdsamp(record_path, channels=[0], sampfrom=sampfrom, sampto="end")
        annotations = wfdb.rdann(record_path, extension="atr", sampfrom=sampfrom, sampto=None)
    else:
        raw_signal= wfdb.rdsamp(record_path, channels=[0], sampfrom=sampfrom, sampto=sampto)
        annotations = wfdb.rdann(record_path, extension="atr", sampfrom=sampfrom, sampto=sampto)

    raw_signal = raw_signal.reshape(-1)
    
    # Filtering
    if use_filter:
        filter_1 = butter_filter(raw_signal, filter_type="highpass", order=3, cutoff_freqs=[1], sampling_freq=annotations.fs)
        filter_2 = butter_filter(filter_1, filter_type="bandstop", order=3, cutoff_freqs=[58, 62], sampling_freq=annotations.fs)
        signal = butter_filter(filter_2, filter_type="lowpass", order=4, cutoff_freqs=[25], sampling_freq=annotations.fs)
    else:
        signal = raw_signal
    
    annotation2sample = list(zip(annotations.symbol, annotations.sample))

    for idx, annot in enumerate(annotation2sample):
        beat_type       = annot[0]    # "N", "V", ... etc.
        r_peak_pos      = annot[1]    # The R peak position
        pulse_start_pos = r_peak_pos - int(length_qrs / 2) + 1    # The sample postion of pulse start (start of QRS)

        # We treat only Normal, VEB, and SVEB signals 
        print(beat_type)
        if beat_type == "N" or beat_type == "S" or beat_type == "V":
            qrs = signal[pulse_start_pos : pulse_start_pos + length_qrs]
            stt = signal[pulse_start_pos + length_qrs + 1 : pulse_start_pos + length_qrs + length_stt]
            #print(qrs.size)
            if qrs.size > 0:
                _, qrs_arcoeffs, _, _, _ = levinson_durbin(qrs, nlags=ar_order_qrs, isacov=False)
            else:
                qrs_arcoeffs = None
            #print(stt.size)  
            if stt.size > 0:
                #print(stt.shape)
                _, stt_arcoeffs, _, _, _ = levinson_durbin(stt, nlags=ar_order_stt, isacov=False)
            else:
                stt_arcoeffs = None

            pre_rr_length  = annotation2sample[idx][1] - annotation2sample[idx - 1][1] if idx > 0 else None
            post_rr_length = annotation2sample[idx + 1][1] - annotation2sample[idx][1] if idx + 1 < annotations.ann_len  else None
            _type = 1 if beat_type == "V" else 0

            """
            beat_dict = OrderedDict([("record", record_path.rsplit(sep="/", maxsplit=1)[-1]), ("type", _type),
                                     ("QRS", qrs), ("ST/T", stt),
                                     ("QRS_ar_coeffs", qrs_arcoeffs), ("ST/T_ar_coeffs", stt_arcoeffs),
                                     ("pre-RR", pre_rr_length), ("post-RR", post_rr_length)])
            """
            beat_list = list()
            beat_list = [("record", record_path.rsplit(sep="/", maxsplit=1)[-1]), ("type", _type), 
                         # ("QRS", qrs), ("ST/T", stt),
                         # ("QRS_ar_coeffs", qrs_arcoeffs), ("ST/T_ar_coeffs", stt_arcoeffs),
                         ("pre-RR", pre_rr_length), ("post-RR", post_rr_length)
                        ]
            #print(qrs_arcoeffs)
            #print('gdlgh')
            for idx, coeff in enumerate(qrs_arcoeffs):
                beat_list.append(("qrs_ar{}".format(idx), coeff))
            print(stt_arcoeffs)
            for idx, coeff in enumerate(stt_arcoeffs):
                beat_list.append(("stt_ar{}".format(idx), coeff))
            
            beat_dict = OrderedDict(beat_list)
            
            qrs_stt_rr_list.append(beat_dict)
    return qrs_stt_rr_list

def series2arCoeffs (series):
    if series.size > 0:
        return np.concatenate(series.tolist()).reshape(series.size, -1)
    else:
        return None


import numpy as np
import pandas as pd

# Check the paper for choosing the lengthes
length_qrs = 40
length_stt = 100

lst = list()

    # Tweak the use_filter param
lst.extend(extract_features("mitdb/" + rec_index, length_qrs, length_stt, ar_order_qrs=3, ar_order_stt=3, use_filter=True))

df = pd.DataFrame(lst)
df.dropna(inplace=True)

y = df["type"].values

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
X = df[df.columns[2:]].values
X = scale(X)




X_test=np.array(X)


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))




#from keras.models import load_model
model = load_model('Trained_model_real.h5')


Y_pred_rnn = model.predict(X_test)


if max(Y_pred_rnn)>.5:
    print('Given patient data is Heart Disease Caution')
    
else:
     print('Given patient data is Normal')


