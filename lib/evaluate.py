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

def evaluate(csv, columns, target, data_name, train_range, test_range, event_start, event_end, n_past= 96, n_future= 12, train_flag= True, predict_flag= True):

    date = datetime.now().strftime("%B_%d_%Y_%H_%M")



    train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_range= train_range, test_range= test_range)#train_test_ratio= 0.8)
    trainX, trainY = ingest.reshape(train_scaled, n_past, n_future)#, timestep_type= "hr")
    testX, testY = ingest.reshape(test_scaled,  n_past, n_future)#, timestep_type= "hr")


    model_names = ['Basic_LSTM', "GRU", 'Stacked_LSTM']#'Bidirectional_LSTM',]

    if train_flag:
        for model_name in model_names:
            print(f'evaluating {model_name}')

            model = models_cuda.prebuilt_models(model_name, trainX, trainY, epochs= 10, batch_size=32, loss= "mse", load_models=False, data_name= data_name)
            validation_loss = models_cuda.evaluate_model(model, testX, testY)
            
            ### JUST TRAIN FOR NOW ###
            models_cuda.plot_model(model_name, validation_loss, data_name)
            K.clear_session()

    if predict_flag:
        _predict(event_start, event_end, model_names, testX, testY, test_dates, data_name)



def _predict(tstart, tend, model_names, testX, testY, test_dates, data_name):

    event_range = [tstart, tend]
    print(event_range, model_names)
    for model_name in model_names:
        print(f"predicting {model_name} over {event_range}")
        predicts = predict.predict(model_name, testX, data_name)
        predict.plot_predicts(model_name, predicts, testY, test_dates, data_name, event_range= event_range, event_plotstep= "Day")
