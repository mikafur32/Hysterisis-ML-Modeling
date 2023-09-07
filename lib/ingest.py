import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates



def read_in(csv, USGS_FLAG= True, renames={}):
    if(USGS_FLAG):
        df = pd.read_csv(csv, low_memory=False)
        df = df.rename(columns= renames)

        df = df[['datetime'] + list(renames.values())]
        df = df.set_index('datetime')
        df.dropna(axis = 0, inplace = True)
    all_dates = df.index.to_series()
    return df, all_dates


def train_test_split(df, train_range, test_range):
    train_from = train_range[0]
    train_to = train_range[1]

    test_from = test_range[0]
    test_to = test_range[1]

    return df[train_from:train_to], df[test_from:test_to]

def ingest(csv, USGS_FLAG= True, renames={}):
    
    df, all_dates = read_in(csv, USGS_FLAG, renames)

    transformed_df = StandardScaler().fit_transform(df)
    transformed_df = pd.DataFrame(transformed_df, columns= list(renames.values()), index=df.index)

    # Split into train and test
    train_range = ['2017-01-01', '2019-12-31']
    test_range = ['2020-01-01', '2020-12-31']
    train_scaled, test_scaled = train_test_split(transformed_df, train_range, test_range)

    # Get train and test timestamps for plotting
    train_dates = train_scaled.index.to_series()
    test_dates = test_scaled.index.to_series()

    train_scaled = train_scaled.to_numpy()
    test_scaled = test_scaled.to_numpy()

    return train_scaled, test_scaled, train_dates, test_dates, all_dates

def reshape(scaled):
    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
    #In this example, the n_features is 3. We will make timesteps = 672 (past 7 days data used for training).

    #Empty lists to be populated using formatted training data
    X = []
    Y = []

    time_to_hr = 4 # 4 timesteps per hour
    time_to_day = time_to_hr * 24 # 24hrs in a day

    n_future = 12 # Number of timesteps we want to look into the future based on the past timesteps. 4 * 3hrs = 12
    n_past = 3 * time_to_day # Number of past timesteps we want to use to predict the future. 7 days x 24hrs x 4 timesteps/hr = 672

    #Reformat input data into a shape: (n_samples x timesteps x n_features)

    for i in range(n_past, len(scaled) - n_future +1):
        X.append(scaled[i - n_past : i, 0:scaled.shape[1]])
        
        Y.append(scaled[i + n_future - 1:i + n_future, 0]) #0 = Discharge

    X, Y = np.array(X), np.array(Y)
    return X, Y



