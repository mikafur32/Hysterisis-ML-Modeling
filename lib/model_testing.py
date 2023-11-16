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
import ingest, predict


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

## FLAGS ##
# RAS model output or USGS
USGS_FLAG = True

### HENRY RAS ###
csv = r"..\data\Henry_WSS_2017_2023.csv"
columns = {'Q': 'Discharge', 'WSS': 'Slope'}

date = datetime.now().strftime("%B_%d_%Y_%H_%M")


target = "Discharge"
data_name = "Henry_RAS_2017_2023_" + f"{date}"


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

train_range = ["1/1/2017 0:00","12/31/2021 23:45"]
test_range = ["1/1/2022 0:00", "12/31/2022 23:45"]

train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, USGS_FLAG=USGS_FLAG, train_range= train_range, test_range= test_range)#train_test_ratio= 0.8)
trainX, trainY = ingest.reshape(train_scaled)#, timestep_type= "hr")
testX, testY = ingest.reshape(test_scaled)#, timestep_type= "hr")


model_names = ['Basic_LSTM', "GRU", 'Bidirectional_LSTM', 'Stacked_LSTM']


for model_name in model_names:
    model = models_cuda.prebuilt_models(model_name, trainX, trainY, epochs= 10, batch_size=32, loss= "nse", load_models=False, data= data_name)
    validation_loss = models_cuda.evaluate_model(model, testX, testY)
    models_cuda.plot_model(model_name, validation_loss, data_name)
    K.clear_session()

'''
tstart = '2022-03-18 00:00:00'
tend = '2022-04-07 00:00:00'
'''

tstart = "3/18/2022 0:00"
tend = "4/7/2022 23:45"
event_range = [tstart, tend]

for model_name in model_names:
    predicts = predict.predict(model_name, testX, data_name)
    predict.plot_predicts(model_name, predicts, testY, test_dates, data_name, event_range= event_range, event_plotstep= "Day")

