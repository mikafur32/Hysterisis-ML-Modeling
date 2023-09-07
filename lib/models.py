from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention

import os

def plot_model(history, model_name, valid_loss):
    plt.title(f"{model_name} Training and Validation Loss")
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Embedded validation loss')
    plt.legend()

    description = f"Validation set loss: {valid_loss}"
    plt.text(0.5, -0.1, description, transform=plt.gca().transAxes,
            fontsize=10, color='gray', ha='center', va='center')
    plt.savefig(f"{model_name}.png")  



def prebuilt_models(model_name, trainX, trainY):
    if(model_name == 'Basic_LSTM'):
        # The LSTM architecture -- Basic RNN | one layer then dense
        model = Sequential()
        model.add(LSTM(units=125, activation="tanh", input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(Dense(units=1))
        # Compiling the model
        model.compile(optimizer="RMSprop", loss="mse")
        model.summary()

    elif(model_name == 'Stacked_LSTM'):
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(64, activation='relu', return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(trainY.shape[1])
        ])
        model.compile(optimizer='adam', loss='mse')
        model.summary()

    
    elif(model_name == 'Attention_LSTM'):
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True),
            SeqSelfAttention(attention_activation='relu'),
            Dropout(0.2),
            Dense(trainY.shape[1])
        ])
        model.compile(optimizer='adam', loss='mse')
        model.summary()
    

    elif(model_name == 'Bidirectional_LSTM'):
        from keras.layers import Bidirectional

        model = Sequential()
        model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[2])))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))

        model.compile(optimizer='adam', loss='mse')
        model.summary()

    elif(model_name == 'GRU'):
        from keras.layers import GRU

        model = Sequential()
        model.add(GRU(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(trainY.shape[1]))
        model.compile(optimizer='adam', loss='mse')



    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1, callbacks=[early_stopping])
    
    plot_model(history, model_name)
    if not "saved_model" in os.listdir(): #If the saved model directory doesn't exist, make it    
        os.makedirs("saved_model")
    model.save(f'saved_model/{model_name}_Saved_Henry_2017_2020')    

    return model, history


def evaluate_model(model, validX, validY):
    validation_loss = model.evaluate(validX, validY, verbose=1)
    print(f'Validation loss: {validation_loss}')



