import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


x = np.array([[1,18],[0,0],[1,16],[0,0]])
y = np.array([1,0,1,0])


layer_1 = keras.layers.Dense(units=4,activation='sigmoid')
layer_2 = keras.layers.Dense(units=1,activation='sigmoid')
model = keras.Sequential(
    [keras.Input(shape=(2,)),
     layer_1,
     layer_2
     ]
)


model.compile(loss = keras.losses.BinaryCrossentropy(),optimizer = keras.optimizers.Adam(learning_rate=0.01))

model.fit(x,y,epochs=100)