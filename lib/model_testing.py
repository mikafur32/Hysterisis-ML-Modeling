import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
import os
from keras import mixed_precision

from datetime import datetime


'''

TODO: memory management!!! 

'''
from keras import backend as K

#os.chdir(".\\lib")
#print(os.getcwd())

#import models_base


import models_cuda
import ingest, predict, evaluate


"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate 13GB of memory on the first GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
"""


policy = mixed_precision.Policy('mixed_float16')

mixed_precision.set_global_policy(
    policy
)



### HENRY RAS ###
csv = r"..\data\Henry_4vars_2017_2023.csv"




train_range = ["1/1/2017 0:00","12/31/2021 23:45"]
test_range = ["1/1/2022 0:00", "12/31/2022 23:45"]
'''
tstart = '2022-03-18 00:00:00'
tend = '2022-04-07 00:00:00'
'''

tstart = "3/18/2022 0:00"
tend = "4/7/2022 23:45"

WSS_V = {"target": "V", "features": { "WSS": "WSS"}, "Name": "WSS_V"}

WSSV_Q = {"target": "Q", "features": { "WSS": "WSS", "V": "V"}, "Name": "WSSV_Q"}

WSSVQ_WL = {"target": "WL", "features": { "WSS": "WSS", "V": "V", "Q": "Q"}, "Name": "WSSVQ_WL"}



tests= [WSS_V, WSSV_Q, WSSVQ_WL]


for test in tests:
    data_name = "Henry_RAS_2017_2023_" + f"{test['Name']}"

    
    evaluate.evaluate(csv, test["features"], test["target"],
                       data_name, train_range, test_range,
                       tstart, tend)

# Tensor board
# tensorboard --logdir=C:\Users\Mikey\Documents\Github\Hysterisis-ML-Modeling\lib\logs

'''
### HENRY ###
csv = r"..\\data\\henry_csv_17-23.csv"
#renames = {'00065': 'Gage Height', '00060': 'Discharge', '72254': 'Velocity'}
columns = {'65': 'Gage Height', '60': 'Discharge', '72254': 'Velocity'}

target = "Discharge"
data_name = "Henry_2017_2020"
'''

'''
csv = "..\\data\\USGS_WS_2017_2023.csv"
columns = {
           "Peoria_WL": "Peoria_WL",
           "Henry_WL": "Henry_WL",
#          "Flow": "Flow",       --- Target
#          "Vel": "Vel",         --- Disregard
           "Slope": "Slope"
           }

target = "Flow"
data_name = "USGS_WS_2017_2023"
'''