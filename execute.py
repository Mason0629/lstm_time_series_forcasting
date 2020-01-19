import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import DataTools_ver_03 as DataTools
from LSTModel import LSTM
import numpy as np
import pandas as pd
import getConfig

Config = {}
Config = getConfig.get_config(config_file='config.ini')

dataset_path = Config["dataset_path"]
df = pd.read_csv(dataset_path)
feature_considered = ['power', 'wind', 'target']
Data = df[feature_considered]
Data = Data.dropna()
data = np.array(Data)

data_split = Config["data_split"]
norm_data, K, B = DataTools.FwdNormal(data, "Standard")
train_data = np.array(norm_data[:data_split]).astype(np.float32)
val_data = np.array(norm_data[data_split:]).astype(np.float32)

time_steps = Config["time_steps"]
pred_steps = Config["pred_steps"]
train_x, train_y = DataTools.GenerateData(train_data, n_inputs=time_steps, pred_step=pred_steps)
val_x, val_y = DataTools.GenerateData(val_data, n_inputs=time_steps, pred_step=pred_steps)


train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_dataset = train_dataset.shuffle(1000).repeat(1).batch(32, drop_remainder=True)
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32).repeat()

units = Config["units"]
n_inputs = Config["n_inputs"]

lstm = LSTM(units, pred_steps)
inputs = tf.keras.Input((time_steps, n_inputs))
outputs = lstm(inputs)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='mae')

def train():
     model.fit(train_dataset, epochs=20, validation_data=val_dataset, validation_steps=10)
        
if __name__ == "__main__":
    train()
    