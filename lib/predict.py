import models, ingest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def predict(model_name, testX):
    model = models.get_model(model_name)
    print("predicting")
    return pd.DataFrame(model.predict(testX, verbose= 1))

def plot_predicts(model_name, predicts, testY, test_dates):
    
    shape = test_dates.shape[0] if predicts.shape[0] > test_dates.shape[0] else predicts.shape[0]

    # Add dates to the sets
    testY = pd.DataFrame(testY, index= test_dates[:shape])
    predicts = predicts.rename({0: "Predicted"}, axis=1)

    plt.figure()
    plt.title(f"{model_name} Predictions")
    plt.plot(pd.to_datetime(testY.index), testY.iloc[:, 0], label='Actual')
    plt.plot(pd.to_datetime(testY.index), predicts.iloc[:, 0], label='Predicted')


    # Format the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Display one tick per month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate the x-axis labels for better visibility

    plt.legend()
    plt.savefig(fr"model_results/{model_name}/{model_name}_predictions.png")  
    plt.close()

