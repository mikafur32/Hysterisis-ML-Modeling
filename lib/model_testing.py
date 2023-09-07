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

os.chdir("lib")
print(os.getcwd())

import models, ingest

## FLAGS ##
# RAS model output or USGS
USGS_FLAG = True

csv = '..\\henry_csv_17-23.csv'
renames = {'65': 'Gage Height', '60': 'Discharge', '72254': 'Velocity'}

train_scaled, test_scaled, train_dates, test_dates, all_dates = ingest.ingest(csv, renames= renames, USGS_FLAG=USGS_FLAG)
trainX, trainY = ingest.reshape(train_scaled)
testX, testY = ingest.reshape(test_scaled)

model_names = ['Basic_LSTM', 'Stacked_LSTM', 'Bidirectional_LSTM', 'Attention_LSTM']
for model_name in model_names:
    model, history = models.prebuilt_models(model_name, trainX, trainY)
    models.evaluate_model(model, testX, testY)
    models.plot_model(history, model_name, model)


