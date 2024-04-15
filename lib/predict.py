
import models_cuda
#import models
import ingest


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np

def predict(model_name, testX, dataname):
    model = models_cuda.get_model(model_name, dataname)
    print("predicting")
    return pd.DataFrame(model.predict(testX, verbose= 1))

def plot_predicts(model_name, predicts, testY, test_dates, dataname, scaler, event_range= None, event_plotstep= "Day"):
    
    testY= testY.flatten()
    predicts = predicts.to_numpy().flatten()

    predicts = pd.concat([pd.Series(predicts),pd.Series(predicts),pd.Series(predicts),pd.Series(predicts)], axis=1)
    testY =  pd.concat([pd.Series(testY),pd.Series(testY),pd.Series(testY),pd.Series(testY)], axis=1)


    predicts = scaler.inverse_transform(predicts)
    testY = scaler.inverse_transform(testY)
    
    # Get the smallest shape
    shape = test_dates.shape[0] if predicts.shape[0] > test_dates.shape[0] else predicts.shape[0]

    testY = pd.DataFrame(testY, index= pd.to_datetime(test_dates[:shape]))

    # Ensure uniform types
    predicts = predicts.astype(np.float64)
    predicts = pd.DataFrame(predicts)

    # Apply datum adjustments
    datum = 425.85
    predicts[0] = predicts[0].apply(lambda x: x - datum)
    print(predicts)

    testY[0]= testY[0].apply(lambda x: x - datum)

    # Export predicts
    if(not os.path.exists(f"lib/model_results/{dataname}/{model_name}/predict_results2")):
        os.makedirs(f"lib/model_results/{dataname}/{model_name}/predict_results2", exist_ok= True) 

    # Set datetime indicies
    predicts["datetime"] = test_dates[:shape].index
    predicts = predicts.set_index("datetime")

    #Evaluate the metrics
    metrics_df = evaluate_metrics(predicts, testY, dataname, model_name)


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

        metrics_df = evaluate_metrics(eventPredicts, eventY, dataname, model_name)

        plt.figure()
        plt.title(f"{model_name} Predictions for event {event_range}")
        plt.plot(pd.to_datetime(eventY.index), eventY.iloc[:, 0], label='Actual')
        plt.plot(pd.to_datetime(eventY.index), eventPredicts.iloc[:, 0], label='Predicted')


        # Format the x-axis
        ax = plt.gca()
        if event_plotstep == "Month":
            ax.xaxis.set_major_locator(mdates.MonthLocator())  # Display one tick per month
        elif event_plotstep == "Week":
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Display one tick per week
        elif event_plotstep == "Day":
            ax.xaxis.set_major_locator(mdates.DayLocator())  # Display one tick per day
        elif event_plotstep == "Hour":
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate the x-axis labels for better visibility
        plt.ylabel("WL (ft)")  
        plt.xlabel("Datetime")  
        plt.legend()
        plt.tight_layout()

        plt.savefig(fr"lib/model_results/{dataname}/{model_name}/predict_results/{model_name}_event_predictions.png")  
        plt.close()

    predicts.to_csv(f"lib/model_results/{dataname}/{model_name}/predict_results/{model_name}_predicts.csv")
    #testY.to_csv(f"lib/model_results/{dataname}/{model_name}/predict_results_flow/{model_name}_event_observed.csv")

    plt.figure()
    plt.title(f"{model_name} Predictions")
    plt.plot(pd.to_datetime(testY.index), testY.iloc[:, 0], label='Observed')
    plt.plot(pd.to_datetime(testY.index), predicts.iloc[:, 0], label='Predicted')

    '''
    # Adding metrics to the plot
    x_pos = 0.05 * len(metrics_df['true'])
    y_pos = plt.ylim()[1] * 0.95
    for index, row in metrics_df.iterrows():
        plt.text(x_pos, y_pos, f"{row['Metric']}: {row['Value']:.2f}", fontsize=9)
        y_pos -= (plt.ylim()[1] - plt.ylim()[0]) * 0.05  
    '''


    # Format the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Display one tick per month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format
    plt.ylabel("WL (ft)")  
    plt.xlabel("Datetime") 
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate the x-axis labels for better visibility
    plt.legend()
    plt.tight_layout()

    plt.savefig(fr"lib/model_results/{dataname}/{model_name}/predict_results/{model_name}_predictions.png")  
    plt.close()



def evaluate_metrics(predicts, y, dataname, model_name):
    import pandas as pd
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.spatial import KDTree
    from permetrics.regression import RegressionMetric

    y,predicts = y[0].to_numpy(), predicts[0].to_numpy()
    evaluator = RegressionMetric(y, predicts)
    kge = evaluator.kling_gupta_efficiency(multi_output="raw_values")
        

    #print(predicts, y)
    mse = np.mean((predicts - y) ** 2)
    rmse = np.sqrt(mse)

    bias = np.mean(predicts - y)

    print(mse,rmse,bias)

 
    true_peaks, _ = find_peaks(y,distance=672)
    predicted_peaks, _ = find_peaks(predicts,distance=672)

   # match peaks based on nearest neighbors
    tree = KDTree(true_peaks.reshape(-1, 1))
    distances, indices = tree.query(predicted_peaks.reshape(-1, 1))

    #print(true_peaks, predicted_peaks)


    if len(true_peaks) > 0 and len(predicted_peaks) > 0:
        mean_peak_error = np.mean(np.abs(y[true_peaks[indices]] - predicts[predicted_peaks]))
        peak_timing_error = np.mean(np.abs(true_peaks[indices] - predicted_peaks))
    else:
        mean_peak_error = None
        peak_timing_error = None

    print(mean_peak_error, peak_timing_error)

    # Create a DataFrame from the results
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'Bias','KGE', 'Mean Peak Error', 'Peak Timing Error'],
        'Value': [mse, rmse, bias, kge, mean_peak_error, peak_timing_error]
    })
    print(results_df)


    results_df.to_csv(f"lib/model_results/{dataname}/{model_name}/{model_name}_metrics.csv")
    return results_df
