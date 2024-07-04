import os, sys, argparse, re
import evaluate, ingest

from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')

mixed_precision.set_global_policy(
    policy
)

"""
=============================================================================
Example CLI Input: 
 
python model_CLI.py -data "C:\\Users\\Mikey\\Documents\\Github\\Hysterisis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv" -model all -train y -train_range "['1/1/2017 0:00','12/31/2021 23:45']" -test_range "['1/1/2022 0:00','12/31/2022 23:45']"  -event_range "['3/18/2022 0:00','4/7/2022 23:45']" -vp y -dn CLITOOLTEST -debug

python model_CLI.py -data "C:\\Users\\Mikey\\Documents\\Github\\Hysterisis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv" -model all -train y -train_range "['1/1/2017 0:00','12/31/2021 23:45']" -test_range "['1/1/2022 0:00','12/31/2022 23:45']"  -event_range "['3/18/2022 0:00','4/7/2022 23:45']" -n_past 96 -n_future 12 -vp y -dn testing2 -debug 

HELP
-h

NO TRAINING
python model_CLI.py -data "C:\\Users\\Mikey\\Documents\\Github\\Hysterisis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv" -model all -train n  -test_range "['1/1/2022 0:00','12/31/2022 23:45']" -n_past 96 -n_future 12 -epochs 1  -event_range
 "['3/18/2022 0:00','4/7/2022 23:45']" -vp y -dn testing2 -debug

=============================================================================
"""




# =============================================================================
# Argument parser, set from the command-line
# =============================================================================

# Get the arguments
parser = argparse.ArgumentParser(description= 
"""\n=============================================================================
\nCLI tool for House & Sison (2024) Hysteretic Stream Forecasting Model\n
=============================================================================""")

parser.add_argument("-data", type=str, required= True,
                    help="The data (csv) to use including the path to folder and extension. See docs for more information about format.")

parser.add_argument("-model", type=str, required= True,
                    help="Supply the name of the model to be ran. See docs for definitions and types. 'all' for all models.")

parser.add_argument("-train", choices= ['y', 'n'], type=str, required= True,
                    help="'y' if need to train a new model. 'n', dataname will be used to load a model.")

parser.add_argument("-train_test_ratio", type= int, required=False, help= "Int range 0-1. Ratio of train to test set size. ie .80 == .8 train and .2 test.")


parser.add_argument("-train_range", type=str, required= False,
                    help= '''
 =============================================================================
 Examples of valid and invalid dates:\n
\n
 Valid: \n
 - "['1/1/2022 0:00','12/31/2022 23:45']"\n
\n
 INVALID\n
 - "['1/1/2022 0:00',\s'12/31/2022 23:45']" - including a space in between the dates\n
 - '["1/1/2022 0:00","12/31/2022 23:45"]' - Invalid quote characters\n
 - "['2022-03-18 00:00:00', '2022-04-07 00:00:00']" - TODO: Invalid datetime format.\n
\n
 =============================================================================\n
'''   )

parser.add_argument("-test_range", type=str, required= False,
                    help="Supply test date ranges: ex \"['1/1/2017 0:00','12/31/2021 23:45']\". If not testing, test_range will be set to event_range for plotting purposes.  ")

parser.add_argument("-n_past", type=str, required= False,
                    help="Supply number of timesteps used in each prediction. Only required if training.")

parser.add_argument("-n_future", type=str, required= False,
                    help="Supply number of timesteps to lag the prediction by.")

parser.add_argument("-epochs", type=int, required= False,
                    help="Number of epochs to train on ")

parser.add_argument("-event_range", type=str, required= True,
                    help="A start and end date for an event to be used for prediction. ex: [2022-03-18 00:00:00', '2022-04-07 00:00:00'] ")

parser.add_argument("-plotstep", type=str, choices=["Day", "Month"],required= False,
                    help= "Time step for plot ticks. Default: Month")

parser.add_argument("-vp", type=str, choices=['y', 'n'], required= False,
                    help="#### Inherent Integration #### \n DEPRECATED: Variable Predictions: 'y' if want predictions from each model in model sequence, 'n' if only WL is required.")

parser.add_argument("-dn", type=str, required= True,
                    help="Supply the name you want for the run.")

parser.add_argument("-scaler", type=str, choices=['y', 'n'], required= False,
                    help="Applies a standard scaler to the input data. Can add options later. ")

parser.add_argument("-debug", action="store_true")


args = parser.parse_args()

# =============================================================================
# Configure the run with command-line arguments
# =============================================================================

# assign and test the arguments
data = args.data
model_names = args.model
train = args.train
train_test_ratio = args.train_test_ratio
train_range = args.train_range
n_past = args.n_past
n_future = args.n_future
test_range = args.test_range
event_range = args.event_range
epochs = args.epochs
#vp = args.vp -- Inherent Integration
plotstep = args.plotstep 
dataname = args.dn
scaler = args.scaler
debug = args.debug


# debug activation
if debug:
    print('''
=============================================================================
 DEBUG STATS
=============================================================================
          ''')
    print("data: ", data)
    print("model_names: ", model_names) 

    print("train: ", train) 
    print("train_test_ratio: ", train_test_ratio) 

    print("train_range: ", train_range) 
    print("n_past: ", n_past) 
    print("n_future: ", n_future) 
    print("epochs: ", epochs) 

    print("test_range: ", test_range) 
    print("event_range: ", event_range) 

    #print("vp: ", vp) 
    print("dataname: ", dataname) 
    print("scaler: ", scaler) 
    print("debug: ", debug)
    print('''
=============================================================================
\n\n
          ''')
# exit if there was no data supplied or the file doesn't exist
if data == None:
  
    print("\nSupply data with the -data command line argument")
    sys.exit()
elif not os.path.exists(data):
    print("\nThe csv doesn't exist, check the path, name, and file type")
    sys.exit()


# Verify correct model name. If 'all' assign model-names to all possible models. 
if model_names == 'all':
    model_names = ['Basic_LSTM', "GRU", 'Stacked_LSTM']

elif not model_names in ['Basic_LSTM', "GRU", 'Stacked_LSTM']:
    print("\n Invalid Model name. See docs for more information.")
    sys.exit()
 

# SEE -h for more information on valid date ranges
valid_date_range_RE = "\['(\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2})','(\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2})'\]" 


# also (train or not)
if train == 'y':
    train=True
else:
    train= False

# also (scale or not)
if scaler == 'y':
    scaler=True
else:
    scaler= False


# Verify test date range
if train and train_range and re.match(valid_date_range_RE, train_range):
    match = re.match(valid_date_range_RE, train_range)
    print(match)
    train_range = [match[1], match[2]]
    train_flag = True

    #print(train_range, type(train_range))
elif os.path.exists(train):
    train = train
    train_flag = False

else:
    raise ValueError("\nInvalid train_range date range format. ex: ['1/1/2017 0:00','12/31/2021 23:45']")


if event_range == 'n':
    event_range = False
elif re.match(valid_date_range_RE, event_range):
    match = re.match(valid_date_range_RE, event_range)
    event_range = [match[1], match[2]]
    #print(event_range, type(event_range))
else:
    raise ValueError("\nInvalid event_range date range format. ex: ['1/1/2017 0:00','12/31/2021 23:45']")

if not test_range:
    test_range = event_range

elif re.match(valid_date_range_RE, test_range):
    match = re.match(valid_date_range_RE, test_range)
    test_range = [match[1], match[2]]
    print(test_range, type(test_range))
else:
    raise ValueError("\nInvalid test_range date range format. ex: ['1/1/2017 0:00','12/31/2021 23:45']")

if train_test_ratio:
    train_range, test_range = None, None


# If add lag in training
lag_flag = False
if n_future and n_past:
    n_future = int(n_future)
    n_past = int(n_past)


    lag_flag = True
   
# Change plotstep if not specified.
if not plotstep:
    plotstep= "Day"

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


# Define tests
tests= [WSSVQ_WL]  #WSS_V, WSSV_Q, ]
tests2= [WSS_V, V_Q, Q_WL]
tests3 =[WSS_WL, WSS_Q]
tests4 = [WL_WL]


if train_flag:
    for test in tests4:
            
        data_name = dataname + f"{test['Name']}"
        print(f"\n=============Running {data_name} =============\n")
        event_start, event_end = event_range[0], event_range[1]
   
        if lag_flag:
            evaluate.evaluate(data,  test["features"], test["target"],
                            data_name, train_range=train_range, test_range=test_range,
                            event_start=event_start, event_end=event_end,n_past=n_past,# epochs=epochs,
                            n_future=n_future, train_flag= train_range,
                            predict_flag= True, plotstep=plotstep, scaler=scaler )
            
        elif train_test_ratio:

            evaluate.evaluate(data,  test["features"], test["target"],
                            data_name, train_range=train_range, test_range=test_range,
                            event_start=event_start, event_end=event_end, train_flag= train_range, epochs=epochs,
                            predict_flag= True, plotstep=plotstep, scaler=scaler)
        
        else:
            evaluate.evaluate(data, test["features"], test["target"],
                            data_name, train_range=train_range, test_range=test_range,
                            event_start=event_start, event_end=event_end, 
                            train_flag= train_range, #epochs=epochs,
                            predict_flag= True, plotstep=plotstep, scaler=scaler)
        
# LATER, get this to choose train/not train in CLI
else:
    event_start, event_end = event_range[0], event_range[1]

    for test in tests4:
        data_name = dataname + f"{test['Name']}"

        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(data, test["target"], train_range= train_range, test_range= test_range)

        if n_past and n_future:   
            testX, testY = ingest.reshape(test_scaled,n_past=n_past, n_future=n_future)
        else:
            testX, testY = ingest.reshape(test_scaled)


        evaluate._predict(event_start, event_end, model_names, testX, testY, test_dates, data_name, plotstep=plotstep)

