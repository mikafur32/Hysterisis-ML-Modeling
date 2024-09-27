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


#policy = mixed_precision.Policy('mixed_float16')

#mixed_precision.set_global_policy(
#    policy
#)

def evaluate(csv, saveto, columns, target, data_name, event_start, event_end, epochs= 1, train_test_ratio= .8, train_range= None, test_range= None, n_past= 96, n_future= 12, train_flag= True, plotstep= "Month", scaler= True):

    date = datetime.now().strftime("%B_%d_%Y_%H_%M")

    if train_range == None or test_range == None:
        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_test_ratio= 0.8)

    else:
        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_range= train_range, test_range= test_range)#train_test_ratio= 0.8)
    


    trainX, trainY = ingest.reshape(train_scaled, n_past, n_future)#, timestep_type= "hr")
    testX, testY = ingest.reshape(test_scaled,  n_past, n_future)#, timestep_type= "hr")

    model_names = ['Basic_LSTM', "GRU", 'Stacked_LSTM']#'Bidirectional_LSTM',]
    

    validation_loss_list = []
    for model_name in model_names:
            if train_flag:

                print(f'evaluating {model_name}')

                # Train
                model = models_cuda.train_models(model_name, trainX, trainY, epochs, batch_size=32, loss= "mse", load_models=False, data_name= data_name)
                
                # Predictions 
                _predict(saveto, event_start, event_end, model_name, testX, testY, test_dates, data_name, plotstep=plotstep, scaler= scaler)

                # Extracting word segments from data_name for validation loss csv  
                def extract_segments(dir_name):
                    parts = dir_name.split("_")
                    bl_part = parts[0] + "_" + parts[1]
                    fl_part = parts[2] + "_" + parts[3][:parts[3].find('WSSVQ')]
                    return bl_part, fl_part
                
                # Validation Loss
                validation_loss = models_cuda.evaluate_model(model, testX, testY)
                seg = extract_segments(data_name)
                validation_loss_list.append([validation_loss, seg[0] ,seg[1] , model_name ])

                # Plotting Validation & Training losses
                models_cuda.plot_model(model_name, validation_loss, data_name)
                K.clear_session()

                # Saving Validation Loss to CSV
                validation_loss_df = pd.DataFrame(validation_loss_list, columns=['Validation Loss', 'BL', 'FL', 'Model Name'])
                csv_path = rf"{saveto}/{data_name}\{model_name}\{model_name}_{data_name}_validation.csv"
                # Create the directory if it doesn't exist (added 7/17)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                validation_loss_df.to_csv(csv_path, index=False)
            else:
                # If already trained, get model & predict
                model = models_cuda.get_model(model_name, saveto=saveto, data=data_name)
                _predict(saveto, event_start, event_end, model_name, testX, testY, test_dates, data_name, plotstep=plotstep, scaler= scaler)


# MAYBE HERE needs to be edited to keep the scaled OBS values for the metrics calc
def _predict(saveto, tstart, tend, model_name, testX, testY, test_dates, data_name, scaler=True, plotstep= "Month"):
    
    event_range = [tstart, tend]
    print(f"predicting {model_name} over {event_range}")

    predicts = predict.predict(model_name, testX, saveto, data_name)
    predict.plot_predicts(saveto, model_name, predicts, testY, test_dates, data_name, scaler=True, event_range= event_range, event_plotstep= plotstep)