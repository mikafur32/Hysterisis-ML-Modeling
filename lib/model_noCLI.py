import os, sys, argparse, re
import evaluate, ingest

from keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')


# =============================================================================
# Configure the run with command-line arguments
# =============================================================================

# assign and test the arguments
data = "C:\\Users\\Mikey\\Documents\\Github\\Hysterisis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv"# args.data
saveto = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\model_results"#args.saveto
model_names = "all" #args.model
train = "y" #args.train
train_range = "['1/1/2017 0:00','12/31/2021 23:45']" #args.train_range
n_past = 4 #args.n_past
n_future = 24#args.n_future
test_range = "['1/1/2022 0:00','12/31/2022 23:45']"#args.test_range
event_range = ['3/18/2022 0:00','4/7/2022 23:45']#args.event_range
#plotstep = "HR" 
dataname = "TEST829BL_1hr_FL_12hr"#args.dn



##### Moved this from above "configure"
# Add a print statement to show raw arguments
print("Raw arguments:", sys.argv)


# =============================================================================
# Run Model
# =============================================================================

# Waterfall LSTM Definitions

WSS_V = {"target": "V", "features": { "WSS": "WSS"}, "Name": "WSS_V"}
WSSV_Q = {"target": "Q", "features": { "WSS": "WSS", "V": "V"}, "Name": "WSSV_Q"}
WSSVQ_WL = {"target": "WL", "features": { "WSS": "WSS", "V": "V", "Q": "Q"}, "Name": "WSSVQ_WL"}


# Other LSTM variations
#WSS_V = {"target": "V", "features": { "WSS": "WSS"}, "Name": "WSS_V"}
V_Q = {"target": "Q", "features": {"V": "V"}, "Name": "V_Q"}
Q_WL = {"target": "WL", "features": {"Q": "Q"}, "Name": "Q_WL"}
WSS_WL = {"target": "WL", "features": {"WSS": "WSS"}, "Name": "WSS_WL"}
WSS_Q = {"target": "Q", "features": {"WSS": "WSS"}, "Name": "WSS_Q"}

WL_WL = {"target": "WL", "features": { "WL":"WL"}, "Name": "Persistence_WL"}


WL_WL = {"target": "WL", "features": { "WL":"WL"}, "Name": "Persistence_WL"}



# Define tests
tests= [WSSVQ_WL]  #WSS_V, WSSV_Q, ]
tests2= [WSS_V, V_Q, Q_WL]
tests3 =[WSS_WL, WSS_Q]



for test in tests:
        
    data_name = dataname + f"{test['Name']}"
    print(f"\n=============Running {data_name} =============\n")
    event_start, event_end = event_range[0], event_range[1]

    evaluate.evaluate(data, saveto, test["features"], test["target"],
                    data_name, train_range=train_range, test_range=test_range,
                    event_start=event_start, event_end=event_end,n_past=n_past,# epochs=epochs,
                    n_future=n_future, train_flag= train_range, #predict_flag= True, 
                    plotstep=None)
              
      
