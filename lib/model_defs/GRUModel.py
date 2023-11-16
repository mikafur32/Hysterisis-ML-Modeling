from model_defs.BaseModel import BaseModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU
class GRUModel(BaseModel):
    def build_model(self):
        self.model = Sequential()
        self.model.add(GRU(128, activation='tanh', input_shape=self.input_shape, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(self.output_units))
        super().compile_model()
