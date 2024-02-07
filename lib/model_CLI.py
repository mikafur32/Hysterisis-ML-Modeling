import os, sys, argparse, re
import evaluate, predict, models_cuda, ingest

from keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')

mixed_precision.set_global_policy(
    policy
)


"""
=============================================================================
Example CLI Input: 
 
python model_CLI.py -data "C:\\Users\\Mikey\\Documents\\Github\\Hysterisis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv" -model all -train y -train_range "['1/1/2017 0:00','12/31/2021 23:45']" -test_range "['1/1/2022 0:00','12/31/2022 23:45']"  -event_range "['3/18/2022 0:00','4/7/2022 23:45']" -vp y -dn CLITOOLTEST -debug

TODO: NO TRAINING
python model_CLI.py -data "C:\\Users\\Mikey\\Documents\\Github\\Hysterisis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv" -model all -train  -train_range n -test_range "['1/1/2022 0:00','12/31/2022 23:45']" -event_range "['3/18/2022 0:00','4/7/2022 23:45']" -vp y -dn CLITOOLTEST -debug


=============================================================================
"""




# =============================================================================
# Argument parser, set from the command-line
# =============================================================================

# Get the arguments
parser = argparse.ArgumentParser(description= 
"""=============================================================================\n
CLI tool for House & Sison (2024) Hysteretic Stream Forecasting Model\n
=============================================================================""")

parser.add_argument("-data", type=str, required= True,
                    help="The data (csv) to use including the path to folder and extension. See docs for more information about format.")

parser.add_argument("-model", type=str, required= True,
                    help="Supply the name of the model to be ran. See docs for definitions and types. 'all' for all models.")

parser.add_argument("-train", type=str, required= True,
                    help="'y' if need to train a new model. Else, supply path to model.")

parser.add_argument("-train_range", type=str, required= True,
                    help= '''
 =============================================================================
 Examples of valid and invalid dates:

 Valid: 
 - "['1/1/2022 0:00','12/31/2022 23:45']"

 INVALID
 - "['1/1/2022 0:00',\s'12/31/2022 23:45']" - including a space in between the dates
 - '["1/1/2022 0:00","12/31/2022 23:45"]' - Invalid quote characters
 - "['2022-03-18 00:00:00', '2022-04-07 00:00:00']" - TODO: Invalid datetime format.

 =============================================================================
'''   )

parser.add_argument("-test_range", type=str, required= True,
                    help="Supply test date ranges: ex \"['1/1/2017 0:00','12/31/2021 23:45']\". 'n' if no test.")

parser.add_argument("-event_range", type=str, required= True,
                    help="A start and end date for an event to be used for prediction. ex: [2022-03-18 00:00:00', '2022-04-07 00:00:00'] ")

parser.add_argument("-vp", type=str, choices=['y', 'n'], required= True,
                    help="Variable Predictions: 'y' if want predictions from each model in model sequence, 'n' if only WL is required.")

parser.add_argument("-dn", type=str, required= True,
                    help="Supply the name you want for the run.")

parser.add_argument("-debug", action="store_true")


args = parser.parse_args()

# =============================================================================
# Configure the run with command-line arguments
# =============================================================================

# assign and test the arguments
data = args.data
model_names = args.model
train = args.train
train_range = args.train_range
test_range = args.test_range
event_range = args.event_range
vp = args.vp
dataname = args.dn
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
    print("train_range: ", train_range) 
    print("test_range: ", test_range) 
    print("event_range: ", event_range) 
    print("vp: ", vp) 
    print("dataname: ", dataname) 
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
else:
    pass

# Verify correct model name. If 'all' assign model-names to all possible models. 
if model_names == 'all':
    model_names = ['Basic_LSTM', "GRU", 'Stacked_LSTM']

elif not model_names in ['Basic_LSTM', "GRU", 'Stacked_LSTM']:
    print("\n Invalid Model name. See docs for more information.")
    sys.exit()
 

# SEE -h for more information on valid date ranges
valid_date_range_RE = "\['(\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2})','(\d{1,2}/\d{1,2}/\d{4}\s\d{1,2}:\d{2})'\]" 


# als
if train == 'y':
    train=True
# Verify validity of path     
elif os.path.exists(train):
    train=train
else:
    print("\nThe path doesn't exist, check the path, name, and file type")
    sys.exit()

# Verify test date range
if train_range == 'n':
    train_range = False

elif train and re.match(valid_date_range_RE, train_range):
    match = re.match(valid_date_range_RE, train_range)
    print(match)
    train_range = [match[1], match[2]]
    
    #print(train_range, type(train_range))

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

if test_range == 'n':
    test_range = False

elif re.match(valid_date_range_RE, test_range):
    match = re.match(valid_date_range_RE, test_range)
    test_range = [match[1], match[2]]
    print(test_range, type(test_range))
else:
    raise ValueError("\nInvalid test_range date range format. ex: ['1/1/2017 0:00','12/31/2021 23:45']")


# =============================================================================
# Run Model
# =============================================================================

# Waterfall LSTM Definitions
WSS_V = {"target": "V", "features": { "WSS": "WSS"}, "Name": "WSS_V"}
WSSV_Q = {"target": "Q", "features": { "WSS": "WSS", "V": "V"}, "Name": "WSSV_Q"}
WSSVQ_WL = {"target": "WL", "features": { "WSS": "WSS", "V": "V", "Q": "Q"}, "Name": "WSSVQ_WL"}

tests= [WSS_V, WSSV_Q, WSSVQ_WL]

if train:
    for test in tests:
        data_name = dataname + f"{test['Name']}"

        event_start, event_end = event_range[0], event_range[1]

        evaluate.evaluate(data, test["features"], test["target"],
                        data_name, train_range, test_range,
                        event_start, event_end, train_range, test_range )
        
else:
    event_start, event_end = event_range[0], event_range[1]

    for test in tests:
        data_name = dataname + f"{test['Name']}"

        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(data, test["target"], train_range= train_range, test_range= test_range)

        testX, testY = ingest.reshape(test_scaled)

        evaluate._predict(event_start, event_end, model_names, testX, testY, test_dates, data_name)




