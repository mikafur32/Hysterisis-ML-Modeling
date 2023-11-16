from model_defs.BaseModel import BaseModel
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
class StackedLSTMModel(BaseModel):
    def build_model(self):
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, activation="tanh", input_shape=self.input_shape),
            LSTM(units=50, activation="tanh"),
            Dense(units=self.output_units)
        ])
        super().compile_model()
