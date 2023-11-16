from model_defs.BaseModel import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.layers import Bidirectional

class BidirectionalLSTMModel(BaseModel):
    def build_model(self):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True), input_shape=self.input_shape))
        self.model.add(LSTM(32, activation='tanh', return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_units))
        super().compile_model()
