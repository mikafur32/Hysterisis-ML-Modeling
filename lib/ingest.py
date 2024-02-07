import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates



def read_in(csv, target, renames={}):

    df = pd.read_csv(csv, low_memory=False)
    df = df.rename(columns= renames)

    # So much work for a oneliner HAHA
    # Reorganizes columns to be "datetime", target, feature values (renames) 
    df =  df[['datetime'] + [target] + list(filter(lambda x: x!=target, list(renames.values())))] if( target in list(renames.values())) else df[['datetime'] +[target] + list(renames.values())]
    
    df = df.set_index('datetime')

    df.dropna(axis = 0, inplace = True)

    # Reorder so that target is first.
    df = df[[target] + [x for x in renames.values() if x != target]]

    all_dates = df.index.to_series()
    return df, all_dates


def train_test_split(df, train_range, test_range):
    train_from = train_range[0]
    train_to = train_range[1]

   
    #print("TEST RANGE",test_range)

    test_from = test_range[0]
    test_to = test_range[1]
    
    #print("TEST RANGE 0", test_range[0])
    #print(test_range[1])


    return df[train_from:train_to], df[test_from:test_to]

def ingest(csv, target, renames={}, train_range= None, test_range= None, train_test_ratio= None):
    
    df, all_dates = read_in(csv, target, renames)


    # If no range is assigned, will use the full range for training & testing
    if(train_range == None or test_range == None):
        train_range, test_range = [all_dates[0], all_dates[-1]], [all_dates[0], all_dates[-1]]
    elif(train_test_ratio):
        train_range  = [all_dates[0], all_dates.iloc[int(np.floor(len(all_dates) * train_test_ratio))]]
        test_range   = [all_dates.iloc[int(np.ceil(len(all_dates) * train_test_ratio))], all_dates[-1]]


    scaler = StandardScaler()
    transformed_df = scaler.fit_transform(df)
    
    # Validate validity of not having all cols in renames. 
    # transformed_df = pd.DataFrame(transformed_df, columns= list(renames.values()), index=df.index)

    transformed_df = pd.DataFrame(transformed_df, columns= df.columns, index=df.index)

    # Split into train and test
    train_scaled, test_scaled = train_test_split(transformed_df, train_range, test_range)

    # Get train and test timestamps for plotting
    train_dates = train_scaled.index.to_series()
    test_dates = test_scaled.index.to_series()

    train_scaled = train_scaled.to_numpy()
    test_scaled = test_scaled.to_numpy()

    return train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler

def reshape(scaled, timestep_type= 'hr'):
    '''

    TODO: Set a parameter for the timestep size. Default set to 4 timesteps per hour.
    
    '''

    X = []
    Y = []

    '''
    Lag shifting test:
        - vary n_future
        - 0 and 100 should change ths distance of orange line start
            - if so, we know that the n_future is working properly as the lag parameter
            - and, larger lag should imply poorer result .:. skipping data
            
    '''

    if(timestep_type == 'hr'):
        time_to_hr = 4 # 4 timesteps per hour
        time_to_day = time_to_hr * 24 # 24hrs in a day

        n_future = 12 # Number of timesteps we want to look into the future based on the past timesteps. 4 * 3hrs = 12
        n_past = 3 * time_to_day # Number of past timesteps we want to use to predict the future. 
    
    
    elif(timestep_type == 'day'):
        n_future = 1 # Number of timesteps we want to look into the future based on the past timesteps. 
        n_past =  7 # Number of past timesteps we want to use to predict the future. 


    #Reformat input data into a shape: (n_samples x timesteps x n_features)

    for i in range(n_past, len(scaled) - n_future + 1):
        X.append(scaled[i - n_past : i, 0:scaled.shape[1]])
        
        ## NOTE: This assumes the target values are the 1st column.

        Y.append(scaled[i + n_future - 1 : i + n_future, 0]) #0 = Discharge 

    X, Y = np.array(X), np.array(Y)
    return X, Y

