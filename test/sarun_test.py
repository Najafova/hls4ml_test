import hls4ml

import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
#from qkeras import *
import tensorflow as tf

# np.random.seed(0)
y=np.random.randn(1)
print("Birinci random ", y)
# y2=np.random.randn(1)
# print(y2)
# tf.random.set_seed(
#     1234
# )

kerasmodel=Sequential()
kerasmodel.add(Dense(2,input_shape=(1,)))
kerasmodel.add(Activation('relu'))
kerasmodel.add(Dense(5))
kerasmodel.add(Activation('relu'))
kerasmodel.compile(optimizer='adam', loss='mse')


hls_model1 = hls4ml.converters.convert_from_keras_model(kerasmodel)
hls_model1.compile()

print("---"*20)
print(kerasmodel.predict(y))
print(hls_model1.predict(y))


y=np.random.randn(1)
print("Ikinci random ", y)

kerasmodel=Sequential()
kerasmodel.add(Dense(2,input_shape=(1,)))
kerasmodel.add(Activation('relu'))
kerasmodel.add(Dense(5))
kerasmodel.add(Activation('relu'))
kerasmodel.compile(optimizer='adam', loss='mse')

hls_model2 = hls4ml.converters.convert_from_keras_model(kerasmodel)
hls_model2.compile()


print("---"*20)
print(kerasmodel.predict(y))
print(hls_model2.predict(y))
