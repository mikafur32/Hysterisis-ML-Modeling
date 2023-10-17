from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention
from keras.initializers import GlorotNormal
import os
import pickle

def plot_model(model_name, valid_loss):

    # Load History
    with open(f'saved_model_multi/trainHistoryDict/{model_name}.pkl', "rb+") as file_pi:
        history = pickle.load(file_pi)

    plt.figure()
    plt.title(f"{model_name} Training and Validation Loss")
    plt.plot(history['loss'], label='Training loss')
    plt.plot(history['val_loss'], label='Embedded validation loss')
    plt.legend()

    description = f"Validation set loss: {valid_loss}."
    plt.text(0.5, -0.1, description, transform=plt.gca().transAxes,
            fontsize=10, color='gray', ha='center', va='center')
    
    os.makedirs(fr"model_results/{model_name}", exist_ok=True) #If the saved model directory doesn't exist, make it    

    plt.savefig(fr"model_results/{model_name}/{model_name}_training_validation.png")  
    plt.close() #Figures are saved rather than printed to kernel

def prebuilt_models(model_name, trainX, trainY, epochs=10, batch_size= 16, load_models=False, location=None, data= None):
    if(load_models):
        return get_model(model_name, data)
    
    if(model_name == 'Basic_LSTM'):
        # The LSTM architecture -- Basic RNN | one layer then dense
        model = Sequential()
        model.add(LSTM(units=125, activation="tanh", input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(units=1))
        # Compiling the model
        model.compile(optimizer="adam", loss="mse")
        model.summary()

    elif(model_name == 'Stacked_LSTM'):
        model = Sequential([
            LSTM(64, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='tanh', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='tanh'),
            Dropout(0.2),
            Dense(trainY.shape[1])
        ])
        model.compile(optimizer='RMSprop', loss='mse')
        model.summary()

    
    elif(model_name == 'Attention_LSTM'):
        initializer = GlorotNormal(seed=42)
        model = Sequential([
            LSTM(64, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True, kernel_initializer=GlorotNormal()),
            SeqSelfAttention(attention_activation='tanh', kernel_initializer=GlorotNormal()),
            Dropout(0.2),
            Dense(trainY.shape[1])
        ])
        model.compile(optimizer='RMSprop', loss='mse')
        model.summary()
    

    elif(model_name == 'Bidirectional_LSTM'):
        from keras.layers import Bidirectional

        model = Sequential()
        model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(LSTM(32, activation='tanh', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))

        model.compile(optimizer='RMSprop', loss='mse')
        model.summary()

    elif(model_name == 'GRU'):
        from keras.layers import GRU

        model = Sequential()
        model.add(GRU(128, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer='RMSprop', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stopping])
    
    # Save model
    os.makedirs("saved_model_multi/trainHistoryDict", exist_ok=True) #If the saved model directory doesn't exist, make it    

    print("saving model")
    model.save(f'saved_model_multi/{model_name}_Saved_Henry_2017_2020')    

    # Save history
    print("saving history") 
    with open(f'saved_model_multi/trainHistoryDict/{model_name}.pkl', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)

    return model


def evaluate_model(model, validX, validY):
    print("evaluating model")
    validation_loss = model.evaluate(validX, validY, verbose=1)
    print(f'Validation loss: {validation_loss}')
    return validation_loss


def get_model(model_name, data= 'Henry_2017_2020'):
    from keras.models import load_model
    print("retrieving and loading model")
    return load_model(f'saved_model_multi/{model_name}_Saved_{data}')
