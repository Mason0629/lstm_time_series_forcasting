import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

class LSTM(tf.keras.Model):
    
    def __init__(self, units, pred_steps):
        super(LSTM, self).__init__()
        self.units = units
        self.lstm_layer = tf.keras.layers.LSTM(self.units)
        self.output_layer = tf.keras.layers.Dense(pred_steps)
        
    def call(self, inputs):
        h = self.lstm_layer(inputs)
        y = self.output_layer(h)
        
        return y
    