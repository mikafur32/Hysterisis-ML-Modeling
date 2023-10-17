import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
import os

os.chdir(".\\lib")
print(os.getcwd())

import models, ingest,predict

## FLAGS ##
# RAS model output or USGS
USGS_FLAG = True


'''
### HENRY ###
csv = r"..\\data\\henry_csv_17-23.csv"
renames = {'00065': 'Gage Height', '00060': 'Discharge', '72254': 'Velocity'}
target = "Discharge"
data_name = "Henry_2017_2020"
'''
csv = "..\\data\\USGS_WS_2017_2023.csv"
renames = {"Peoria date" : "datetime"}
target = "Flow"
data_name = "USGS_WS_2017_2023"

train_scaled, test_scaled, train_dates, test_dates, all_dates = ingest.ingest(csv, target, renames= renames, USGS_FLAG=USGS_FLAG)
                                                                               #train_range= ['2017-01-01', '2017-02-31'], test_range= ['2020-01-01', '2020-01-31'])
trainX, trainY = ingest.reshape(train_scaled)#, timestep_type= "hr")
testX, testY = ingest.reshape(test_scaled)#, timestep_type= "hr")

model_names = ['Basic_LSTM', "GRU", 'Bidirectional_LSTM','Stacked_LSTM']
for model_name in model_names:
    model = models.prebuilt_models(model_name, trainX, trainY, epochs= 10, batch_size=16, load_models=False, data= data_name)
    validation_loss = models.evaluate_model(model, testX, testY)
    models.plot_model(model_name, validation_loss)
    predicts = predict.predict(model_name, testX)
    predict.plot_predicts(model_name, predicts, testY, test_dates)


