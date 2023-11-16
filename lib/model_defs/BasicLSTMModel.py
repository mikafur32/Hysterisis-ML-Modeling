from model_defs.BaseModel import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


class BasicLSTMModel(BaseModel):
    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=125, activation="tanh", input_shape=self.input_shape))
        self.model.add(Dense(units=self.output_units))
        super().compile_model()
