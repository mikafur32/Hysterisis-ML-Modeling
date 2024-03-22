
import models_cuda
#import models
import ingest

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np




def predict(model_name, testX, dataname):
    model = models_cuda.get_model(model_name, dataname)
    print("predicting")
    print(model.input_shape)
    print(testX.shape)
    return pd.DataFrame(model.predict(testX, verbose= 1))

def plot_predicts(model_name, predicts, testY, test_dates, dataname, event_range= None, event_plotstep= "Month"):
    
    '''
    TODO:
    1. Inverse Transform
    2. Y-axis labels

    '''
    # Get the smallest shape
    shape = test_dates.shape[0] if predicts.shape[0] > test_dates.shape[0] else predicts.shape[0]

    # Add dates to the sets
    testY = pd.DataFrame(testY, index= pd.to_datetime(test_dates[:shape]))

    # Ensure uniform types
    predicts = predicts.astype(np.float64)
    predicts = pd.DataFrame(predicts)

    # Calc MSE values
    mse_value = tf.keras.metrics.mean_squared_error(testY,predicts)

    # Set datetime indicies
    predicts["datetime"] = test_dates[:shape].index
    predicts = predicts.set_index("datetime")



    if not ((event_range[0] in test_dates) or (event_range[1] in test_dates)):
        raise ValueError(f"Event, {event_range}, not in test set daterange. Please choose another range.")
        
    if event_range != None:
        # Current format does not require pd.to_datetime, may change with different inputs.
        #event_range = pd.to_datetime(event_range)
        t_start = str(event_range[0])
        t_end = str(event_range[1])

        eventY = testY.loc[t_start : t_end]
        print(t_start, t_end)
        eventPredicts = predicts.loc[t_start : t_end]

        plt.figure()
        plt.title(f"{model_name} Predictions for event: {dataname}")
        plt.plot(pd.to_datetime(eventY.index), eventY.iloc[:, 0], label='Actual')
        plt.plot(pd.to_datetime(eventY.index), eventPredicts.iloc[:, 0], label='Predicted')


        # Format the x-axis
        ax = plt.gca()
        if event_plotstep == "Month":
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Display one tick per month
        elif event_plotstep == "Day":
            ax.xaxis.set_major_locator(mdates.DayLocator())  # Display one tick per day

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate the x-axis labels for better visibility

        plt.legend()
        plt.savefig(fr"model_results/{dataname}/{model_name}/{model_name}_event_predictions.png")  
        plt.close()


    plt.figure()
    plt.title(f"{model_name} Predictions")
    plt.text(0.8, 1, f'MSE: {mse_value}', fontsize=14, ha='center')
    plt.plot(pd.to_datetime(testY.index), testY.iloc[:, 0], label='Observed')
    plt.plot(pd.to_datetime(testY.index), predicts.iloc[:, 0], label='Simulated')


    # Format the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Display one tick per month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate the x-axis labels for better visibility

    plt.legend()
    plt.savefig(fr"model_results/{dataname}/{model_name}/{model_name}_predictions.png")  
    plt.close()



