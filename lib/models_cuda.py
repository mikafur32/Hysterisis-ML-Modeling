from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import GlorotNormal
import tensorflow.keras.backend as K
import os
import pickle

from model_defs.BaseModel import BaseModel
from model_defs.BasicLSTMModel import BasicLSTMModel
from model_defs.StackedLSTMModel import StackedLSTMModel
from model_defs.BidirectionalLSTMModel import BidirectionalLSTMModel
from model_defs.GRUModel import GRUModel
from tensorflow.keras.models import load_model

'''
TODO: 
1. Feature importances for each
'''

def nseloss(y_true, y_pred):
  return K.sum((y_pred-y_true)**2)/K.sum((y_true-K.mean(y_true))**2)


def plot_model(model_name, valid_loss, data):

    # Load History
    with open(f'saved_model_multi/{data}/trainHistoryDict/{model_name}.pkl', "rb+") as file_pi:
        history = pickle.load(file_pi)

    plt.figure()
    plt.title(f"{model_name} Training and Validation Loss")
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Embedded validation loss')
    plt.legend()

    description = f"Validation set loss: {valid_loss}."
    plt.text(0.5, -0.1, description, transform=plt.gca().transAxes,
            fontsize=10, color='gray', ha='center', va='center')
    
    os.makedirs(fr"model_results/{data}/{model_name}", exist_ok=True) #If the saved model directory doesn't exist, make it    

    plt.savefig(fr"model_results/{data}/{model_name}/{model_name}_training_validation.png")  
    plt.close() #Figures are saved rather than printed to kernel




def prebuilt_models(model_name, trainX, trainY, epochs=10, batch_size=16, loss="mse", load_models=False, data=None):
    if(load_models):
        print("retrieving and loading model")
        return load_model(f'saved_model_multi/{data}/{model_name}_Saved_{data}')

    # Depending on the loss, we might need to pass a custom loss function
    if loss == "nse":
        loss = nseloss  

    model = None
    output_units = trainY.shape[1]

    # Instantiate the appropriate model class
    if(model_name == 'Basic_LSTM'):
        model = BasicLSTMModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss)
        
    elif(model_name == 'Stacked_LSTM'):
        model = StackedLSTMModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss)
    
    elif(model_name == 'Bidirectional_LSTM'):
        model = BidirectionalLSTMModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss)
    
    elif(model_name == 'GRU'):
        model = GRUModel(input_shape=(trainX.shape[1], trainX.shape[2]), output_units=output_units, loss=loss)


    # Save model and history
    model_directory = f"saved_model_multi/{data}"
    

    # Build and compile the model
    if model is not None:
        model.build_model()

    # Train the model
    history = model.train_model(trainX, trainY, epochs=epochs, batch_size=batch_size, checkpoint_path= model_directory)
    
   
    print("saving model")
    model.model.save(f'{model_directory}\\{model_name}_Saved_{data}')    
    print("saving history") 

    os.makedirs(f"{model_directory}/trainHistoryDict", exist_ok=True) #If the saved model directory doesn't exist, make it    
    with open(f'{model_directory}/trainHistoryDict/{model_name}.pkl', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)

    return model.model



def evaluate_model(model, validX, validY):
    print("evaluating model")
    validation_loss = model.evaluate(validX, validY, verbose=1)
    print(f'Validation loss: {validation_loss}')
    return validation_loss


def get_model(model_name, data= 'Henry_2017_2020'):
    from keras.models import load_model
    print("retrieving and loading model")
    return load_model(f'saved_model_multi/{data}/{model_name}_Saved_{data}')
