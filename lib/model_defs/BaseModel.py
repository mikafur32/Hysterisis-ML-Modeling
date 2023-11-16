
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import GlorotNormal

class BaseModel:
    def __init__(self, input_shape, output_units, loss='mse'):
        self.input_shape = input_shape
        self.output_units = output_units
        self.loss = loss
        self.model = None

    def build_model(self):
        pass

    def compile_model(self):
        self.model.compile(optimizer="RMSprop", loss=self.loss)

    def train_model(self, trainX, trainY, epochs=10, batch_size=16):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1, callbacks=[early_stopping])
        return history
