import pytest
import hls4ml
import tensorflow as tf 
import numpy as np
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Conv2D, \
                                    Reshape, ELU, LeakyReLU, ThresholdedReLU, \
                                    PReLU, BatchNormalization, Add, Subtract, \
                                    Multiply, Average, Maximum, Minimum, \
                                    MaxPooling1D, MaxPooling2D, AveragePooling1D, \
                                    AveragePooling2D
                                    
# Input, ***Reshape, 
# ***Dense, BinaryDense, TernaryDense
# ***Activation, ***LeakyReLU, ***ThresholdedReLU, ***ELU, ***PReLU
# BatchNormalization
# ***Conv1D, ***Conv2D
# Merge layers - Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate
# MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D


# ALMOST DONE
# TODO Consider BinaryDense ve TernaryDense layers
def test_dense():
    model = tf.keras.models.Sequential()
    model.add(Dense(64, 
              input_shape=(16,), 
              name='Dense', 
              use_bias=True,
              kernel_initializer='lecun_uniform',
            #   kernel_quantizer= "quantized_bits(1,2,3,4,5,6,7,8,0,1,2,3,1)",
              bias_initializer='zeros', 
              kernel_regularizer=None,
              bias_regularizer=None,
              activity_regularizer=None, 
              kernel_constraint=None, 
              bias_constraint=None))
    model.add(Activation(activation='elu', name='Activation'))
    model.compile(optimizer='adam', loss='mse')

    hls_model = hls4ml.converters.convert_from_keras_model(model)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes["class_name"] == model.layers[0]._name
    assert list(hls_model.get_layers())[2].attributes['class_name'] == model.layers[1]._name
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0].input_shape[1:])
    assert list(hls_model.get_layers())[1].attributes['n_in'] == model.layers[0].input_shape[1:][0]
    assert list(hls_model.get_layers())[1].attributes['n_out'] == model.layers[0].output_shape[1:][0]
    assert list(hls_model.get_layers())[2].attributes['activation'] == str(model.layers[1].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]


#DONE
def test_reshape():
  model = tf.keras.models.Sequential()
  model.add(Reshape((3,4), input_shape=(12,)))
  model.add(Activation(activation="elu", name='Activation'))
  model.compile(optimizer='adam', loss='mse')

  hls_model = hls4ml.converters.convert_from_keras_model(model)
  assert len(model.layers) + 1 == len(hls_model.get_layers())
  assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
  assert list(hls_model.get_layers())[1].attributes["name"] == model.layers[0]._name
  assert list(hls_model.get_layers())[2].attributes['class_name'] == model.layers[1]._name

  assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0].input_shape[1:])
  assert list(hls_model.get_layers())[1].attributes['target_shape'] == list(model.layers[0].target_shape)
  assert list(hls_model.get_layers())[2].attributes['activation'] == str(model.layers[1].activation).split()[1]


# DONE 
keras_activation_functions = [LeakyReLU, ELU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_leakyrelu_elu(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64, 
              input_shape=(16,), 
              name='Dense', 
              kernel_initializer='lecun_uniform', 
              kernel_regularizer=None))
    model.add(activation_functions(alpha=1.0))

    hls_model = hls4ml.converters.convert_from_keras_model(model)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[0]._name

    if activation_functions == 'ELU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ELU'
    elif activation_functions == 'LeakyReLU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'LeakyReLU'

    assert list(hls_model.get_layers())[0].attributes['input_shape'][0] == model.layers[0]._batch_input_shape[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes["n_in"] == model.layers[0]._batch_input_shape[1]  
    assert list(hls_model.get_layers())[1].attributes["n_out"] == list(model.layers[0].output_shape)[1] 


# DONE
keras_activation_functions = [ThresholdedReLU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_thresholdedrelu(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64, 
              input_shape=(16,), 
              name='Dense', 
              kernel_initializer='lecun_uniform', 
              kernel_regularizer=None))
    model.add(activation_functions(theta=1.0))

    hls_model = hls4ml.converters.convert_from_keras_model(model)

    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[0]._name

    if activation_functions == 'ThresholdedReLU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ThresholdedReLU'

    assert list(hls_model.get_layers())[0].attributes['input_shape'][0] == model.layers[0]._batch_input_shape[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes["n_in"] == model.layers[0]._batch_input_shape[1]  
    assert list(hls_model.get_layers())[1].attributes["n_out"] == list(model.layers[0].output_shape)[1] 


# DONE
keras_activation_functions = [PReLU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_prelu(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64, 
              input_shape=(16,), 
              name='Dense', 
              kernel_initializer='lecun_uniform', 
              kernel_regularizer=None))
    model.add(activation_functions(alpha_initializer="zeros",))

    hls_model = hls4ml.converters.convert_from_keras_model(model)
    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[0]._name

    if activation_functions == 'PReLU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'PReLU'

    assert list(hls_model.get_layers())[0].attributes['input_shape'][0] == model.layers[0]._batch_input_shape[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes["n_in"] == model.layers[0]._batch_input_shape[1]  
    assert list(hls_model.get_layers())[1].attributes["n_out"] == list(model.layers[0].output_shape)[1] 


# DONE
keras_activation_functions = [Activation]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation(activation_functions):
    model = tf.keras.models.Sequential()
    model.add(Dense(64, 
              input_shape=(16,), 
              name='Dense', 
              kernel_initializer='lecun_uniform', 
              kernel_regularizer=None))
    model.add(Activation(activation='relu', name='Activation'))

    hls_model = hls4ml.converters.convert_from_keras_model(model)
    assert len(model.layers) + 1 == len(hls_model.get_layers())
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes['class_name'] == model.layers[0]._name

    if activation_functions == 'Activation':
      assert list(hls_model.get_layers())[2].attributes["activation"] == str(model.layers[1].activation).split()[1]

    assert list(hls_model.get_layers())[0].attributes['input_shape'][0] == model.layers[0]._batch_input_shape[1]
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes["n_in"] == model.layers[0]._batch_input_shape[1]  
    assert list(hls_model.get_layers())[1].attributes["n_out"] == list(model.layers[0].output_shape)[1] 


# DONE
keras_conv1d = [Conv1D]
@pytest.mark.parametrize("conv1d", keras_conv1d)
def test_conv1d(conv1d):
    model = tf.keras.models.Sequential()
    input_shape = (4, 10, 128)
    model.add(conv1d(32, 3, 
                     input_shape=input_shape[1:]))
    model.add(Activation(activation='relu'))

    hls_model = hls4ml.converters.convert_from_keras_model(model)

    assert len(model.layers) + 1 == len(hls_model.get_layers()) 
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name

    if conv1d == 'Conv1D':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'Conv1D'
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0]._batch_input_shape[1:])
    assert list(hls_model.get_layers())[1].attributes['filt_width'] == model.layers[0].kernel_size[0]
    assert list(hls_model.get_layers())[1].attributes['n_filt'] == model.layers[0].filters
    assert list(hls_model.get_layers())[1].attributes["n_in"] == model.layers[0]._batch_input_shape[1] 
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes["n_out"] == list(model.layers[0].output_shape)[1] 
    assert list(hls_model.get_layers())[2].attributes["activation"] == str(model.layers[1].activation).split()[1]


# DONE
keras_conv2d = [Conv2D]
@pytest.mark.parametrize("conv2d", keras_conv2d)
def test_conv2d(conv2d):
    model = tf.keras.models.Sequential()
    input_shape = (4, 28, 28, 3)
    model.add(conv2d(2,12, 
                     input_shape=input_shape[1:]))
    model.add(Activation(activation='relu'))

    hls_model = hls4ml.converters.convert_from_keras_model(model)

    assert len(model.layers) + 1 == len(hls_model.get_layers()) 
    assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name

    if conv2d == 'Conv2D':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'Conv2D'
    assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0]._batch_input_shape[1:])
    assert list(hls_model.get_layers())[1].attributes['filt_width'] == model.layers[0].kernel_size[0]
    assert list(hls_model.get_layers())[1].attributes['n_filt'] == model.layers[0].filters
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[2].attributes['activation'] == str(model.layers[1].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes['data_format'] == model.layers[0].data_format
    assert list(hls_model.get_layers())[1].attributes['in_height'] == model.layers[0]._batch_input_shape[2]
    assert list(hls_model.get_layers())[1].attributes['in_width'] == model.layers[0]._batch_input_shape[1]
    assert list(hls_model.get_layers())[1].attributes['n_filt'] == model.layers[0].filters



# # TODO batch key error
# def test_input():
#   model = tf.keras.models.Sequential()
#   model.add(tf.keras.layers.InputLayer(input_shape=(4,)))
#   model.add(Dense(64, 
#               name='Dense', 
#               kernel_initializer='lecun_uniform', 
#               kernel_regularizer=None))
#   model.add(Activation(activation="elu", name='Activation'))
#   # model.compile(optimizer='adam', loss='mse')
  
#   hls_model = hls4ml.converters.convert_from_keras_model(model)

# test_input()

# TODO
# keras_normalization_function = [BatchNormalization]
# @pytest.mark.parametrize("normalization_function", keras_normalization_function)
# def test_batch_normalization:
#   model = tf.keras.models.Sequential()