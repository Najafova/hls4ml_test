import pytest
import hls4ml
import tensorflow as tf 
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, Conv1D, Conv2D, \
                                    Reshape, ELU, LeakyReLU, ThresholdedReLU, \
                                    PReLU, BatchNormalization, Add, Subtract, \
                                    Multiply, Average, Maximum, Minimum, Concatenate, \
                                    MaxPooling1D, MaxPooling2D, AveragePooling1D, \
                                    AveragePooling2D
import math                       
# ***Input, ***Reshape, 
# ***Dense, BinaryDense, TernaryDense
# ***Activation, ***LeakyReLU, ***ThresholdedReLU, ***ELU, ***PReLU
# ~BatchNormalization
# ***Conv1D, ***Conv2D
# Merge layers - Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate
# ***MaxPooling1D, ***MaxPooling2D, ***AveragePooling1D, ***AveragePooling2D

# TODO there are too much assertion functions. replace them with helper functions
# TODO write models with functional model too
# TODO the other stages of conversion; maybe 4 stages
# TODO save the models in h5py file 
# TODO C++ test
# TODO JSON codes 
# TODO Focus on Convolutional layers mor
# TODO Convert from h5py file and yml file 


# ALMOST DONE
# TODO Consider BinaryDense ve TernaryDense layers
def test_dense():
    model = tf.keras.models.Sequential()
    model.add(Dense(64, 
              input_shape=(16,), 
              name='Dense', 
              use_bias=True,
              kernel_initializer='lecun_uniform',
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
  assert list(hls_model.get_layers())[1].attributes["name"] == model.layers[0]._name
  assert list(hls_model.get_layers())[1].attributes['target_shape'] == list(model.layers[0].target_shape)


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

    if activation_functions == 'ELU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ELU'
    elif activation_functions == 'LeakyReLU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'LeakyReLU'


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

    if activation_functions == 'ThresholdedReLU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ThresholdedReLU'


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

    if activation_functions == 'PReLU':
      assert list(hls_model.get_layers())[2].attributes['class_name'] == 'PReLU'


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

    if activation_functions == 'Activation':
      assert list(hls_model.get_layers())[2].attributes["activation"] == str(model.layers[1].activation).split()[1] 


# DONE
keras_conv1d = [Conv1D]
padds_options = ['same', 'valid']
@pytest.mark.parametrize("conv1d", keras_conv1d)
@pytest.mark.parametrize("padds", padds_options)
def test_conv1d(conv1d, padds):
    model = tf.keras.models.Sequential()
    input_shape = (10, 128, 4)
    model.add(conv1d(filters=32, 
                     kernel_size=3, 
                     strides=1, 
                     padding=padds, 
                     activation='relu', 
                     input_shape=input_shape[1:], 
                     kernel_initializer='normal', 
                     use_bias=False,
                     data_format='channels_last'))
    model.add(Activation(activation='relu'))

    hls_model = hls4ml.converters.convert_from_keras_model(model)
 
    assert len(model.layers) + 2 == len(hls_model.get_layers()) 
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name
 
    if conv1d == 'Conv1D':
      assert list(hls_model.get_layers())[1].attributes['class_name'] == 'Conv1D'
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes["n_in"] == model.layers[0]._batch_input_shape[1]
    assert list(hls_model.get_layers())[1].attributes['filt_width'] == model.layers[0].kernel_size[0]
    assert list(hls_model.get_layers())[1].attributes['n_chan'] == model.layers[0].input_shape[2]
    assert list(hls_model.get_layers())[1].attributes['n_filt'] == model.layers[0].filters
    assert list(hls_model.get_layers())[1].attributes['stride'] == model.layers[0].strides[0]
    assert list(hls_model.get_layers())[1].attributes['padding'] == model.layers[0].padding
    assert list(hls_model.get_layers())[1].attributes['data_format'] == model.layers[0].data_format
    assert list(hls_model.get_layers())[1].attributes["n_out"] == list(model.layers[0].output_shape)[1] 
 
    out_width	= math.ceil(float(model.layers[0]._batch_input_shape[2]) / float(model.layers[0].strides[0]))
    pad_along_width	= max((out_width - 1) * model.layers[0].strides[0] + model.layers[0].kernel_size[0] - model.layers[0]._batch_input_shape[2], 0)
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    out_valid	= math.ceil(float(model.layers[0]._batch_input_shape[1] - model.layers[0].kernel_size[0] + 1) / float(model.layers[0].strides[0]))
    
    if model.layers[0].padding == 'same':
        assert list(hls_model.get_layers())[1].attributes['pad_left'] == pad_left
        assert list(hls_model.get_layers())[1].attributes['pad_right'] == pad_right
        
    elif model.layers[0].padding == 'valid':
        assert list(hls_model.get_layers())[1].attributes['pad_left'] == 0
        assert list(hls_model.get_layers())[1].attributes['pad_right'] == 0
    

# DONE
keras_conv2d = [Conv2D]
padds_options = ['same', 'valid']
chans_options = ['channels_first', 'channels_last']
@pytest.mark.parametrize("conv2d", keras_conv2d)
@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
def test_conv2d(conv2d, chans, padds):
    model = tf.keras.models.Sequential()
    input_shape = (4, 4, 28, 30)
    model.add(conv2d(filters=32, 
                     kernel_size=(4,4), 
                     strides=(4,4), 
                     padding=padds, 
                     activation='relu', 
                     input_shape=input_shape[1:], 
                     kernel_initializer='normal', 
                     use_bias=False,
                     data_format=chans
                     ))
    model.add(Activation(activation='relu'))

    hls_model = hls4ml.converters.convert_from_keras_model(model)

    assert len(model.layers) + 2 == len(hls_model.get_layers()) 
    assert list(hls_model.get_layers())[1].attributes['name'] == model.layers[0]._name

    if conv2d == 'Conv2D':
      assert list(hls_model.get_layers())[1].attributes['class_name'] == 'Conv2D'
    assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]
    assert list(hls_model.get_layers())[1].attributes['filt_width'] == model.layers[0].kernel_size[1]
    assert list(hls_model.get_layers())[1].attributes['filt_height'] == model.layers[0].kernel_size[0]
    assert list(hls_model.get_layers())[1].attributes['n_filt'] == model.layers[0].filters
    assert list(hls_model.get_layers())[1].attributes['stride_width'] == model.layers[0].strides[1]
    assert list(hls_model.get_layers())[1].attributes['stride_height'] == model.layers[0].strides[0]
    assert list(hls_model.get_layers())[1].attributes['padding'] == model.layers[0].padding
    assert list(hls_model.get_layers())[1].attributes['data_format'] == model.layers[0].data_format

    if model.layers[0].data_format == 'channels_first':
      assert list(hls_model.get_layers())[1].attributes['n_chan'] == model.layers[0]._batch_input_shape[1]
      assert list(hls_model.get_layers())[1].attributes['in_height'] == model.layers[0]._batch_input_shape[2]
      assert list(hls_model.get_layers())[1].attributes['in_width'] == model.layers[0]._batch_input_shape[3]
      assert list(hls_model.get_layers())[1].attributes['out_height'] == model.layers[0].output_shape[2]
      assert list(hls_model.get_layers())[1].attributes['out_width'] == model.layers[0].output_shape[3]
    elif model.layers[0].data_format == 'channels_last':
      assert list(hls_model.get_layers())[1].attributes['n_chan'] == model.layers[0]._batch_input_shape[3]
      assert list(hls_model.get_layers())[1].attributes['in_height'] == model.layers[0]._batch_input_shape[1]
      assert list(hls_model.get_layers())[1].attributes['in_width'] == model.layers[0]._batch_input_shape[2]
      assert list(hls_model.get_layers())[1].attributes['out_height'] == model.layers[0].output_shape[1]
      assert list(hls_model.get_layers())[1].attributes['out_width'] == model.layers[0].output_shape[2]

    if model.layers[0].padding =='same':
      if model.layers[0].data_format == 'channels_first':
        out_height	= model.layers[0].output_shape[2]
        out_width	= model.layers[0].output_shape[3]
        pad_along_height	= max((out_height - 1) * model.layers[0].strides[0] + model.layers[0].kernel_size[0] - model.layers[0]._batch_input_shape[2], 0)
        pad_along_width	= max((out_width - 1) * model.layers[0].strides[1] + model.layers[0].kernel_size[1] - model.layers[0]._batch_input_shape[3], 0)

      elif model.layers[0].data_format == 'channels_last':
        out_height	= model.layers[0].output_shape[1]
        out_width	= model.layers[0].output_shape[2]
        pad_along_height	= max((out_height - 1) * model.layers[0].strides[0] + model.layers[0].kernel_size[0] - model.layers[0]._batch_input_shape[1], 0)
        pad_along_width	= max((out_width - 1) * model.layers[0].strides[1] + model.layers[0].kernel_size[1] - model.layers[0]._batch_input_shape[2], 0)
      
      pad_top	= pad_along_height // 2
      pad_bottom	= pad_along_height - pad_top
      pad_left	= pad_along_width // 2
      pad_right	= pad_along_width - pad_left
      assert list(hls_model.get_layers())[1].attributes['pad_top'] == pad_top
      assert list(hls_model.get_layers())[1].attributes['pad_bottom'] == pad_bottom
      assert list(hls_model.get_layers())[1].attributes['pad_left'] == pad_left
      assert list(hls_model.get_layers())[1].attributes['pad_right'] == pad_right

    elif model.layers[0].padding =='valid':
      assert list(hls_model.get_layers())[1].attributes['pad_top'] == 0
      assert list(hls_model.get_layers())[1].attributes['pad_bottom'] == 0
      assert list(hls_model.get_layers())[1].attributes['pad_left'] == 0
      assert list(hls_model.get_layers())[1].attributes['pad_right'] == 0


# DONE
pooling_layers = [MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D]
padds_options = ['same', 'valid']
chans_options = ['channels_first', 'channels_last']
@pytest.mark.parametrize("poolings", pooling_layers)
@pytest.mark.parametrize("padds", padds_options)
@pytest.mark.parametrize("chans", chans_options)
def test_pooling(poolings, padds, chans):
    model = tf.keras.models.Sequential()
    if poolings == 'MaxPooling2D' or poolings == 'AveragePooling2D':
        model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        hls_model = hls4ml.converters.convert_from_keras_model(model)

        for i in range(2):
          assert list(hls_model.get_layers())[i + 3].attributes['name'] == model.layers[i + 1]._name
          assert list(hls_model.get_layers())[i + 3].attributes['class_name'][-2] == str(2)
          assert list(hls_model.get_layers())[i + 3].attributes['stride_height'] == model.layers[i + 1].strides[0]
          assert list(hls_model.get_layers())[i + 3].attributes['stride_width'] == model.layers[i + 1].strides[1]
          assert list(hls_model.get_layers())[i + 3].attributes['pool_height'] == model.layers[i + 1].pool_size[1]
          assert list(hls_model.get_layers())[i + 3].attributes['pool_width'] == model.layers[i + 1].pool_size[0]
          assert list(hls_model.get_layers())[i + 3].attributes['padding'] == model.layers[i + 1].padding

          if list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_last':
            assert list(hls_model.get_layers())[i + 3].attributes['in_height'] == model.layers[i + 1].input_shape[1]
            assert list(hls_model.get_layers())[i + 3].attributes['in_width'] == model.layers[i + 1].input_shape[2]
            assert list(hls_model.get_layers())[i + 3].attributes['n_filt'] == model.layers[i + 1].input_shape[3]
          elif list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_first':
            assert list(hls_model.get_layers())[i + 3].attributes['in_height'] == model.layers[i + 1].input_shape[2]
            assert list(hls_model.get_layers())[i + 3].attributes['in_width'] == model.layers[i + 1].input_shape[3]
            assert list(hls_model.get_layers())[i + 3].attributes['n_filt'] == model.layers[i + 1].input_shape[1]

          if list(hls_model.get_layers())[i + 3].attributes['padding'] == 'same':
            # Height
            in_height = model.layers[i + 1].input_shape[1]
            if model.layers[i + 1].data_format == 'channels_first':
              in_height = model.layers[i + 1].input_shape[2]
            out_height = int(math.ceil(float(in_height) / float(model.layers[i + 1].strides[0])))
            assert out_height == list(hls_model.get_layers())[i + 3].attributes['out_height']
            if in_height % model.layers[i + 1].strides[0] == 0:
              pad_along_height = max(model.layers[i + 1].pool_size[1] - model.layers[i + 1].strides[0], 0)
            else:
              pad_along_height = max(model.layers[i + 1].pool_size[1] - (in_height % model.layers[i + 1].strides[0]), 0)
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            assert pad_bottom == list(hls_model.get_layers())[i + 3].attributes['pad_bottom']
            assert pad_top == list(hls_model.get_layers())[i + 3].attributes['pad_top']
            
            # Width
            in_width = model.layers[i + 1].input_shape[2]
            if model.layers[i + 1].data_format == 'channels_first':
              in_height = model.layers[1].input_shape[i + 3]
            out_width = int(math.ceil(float(in_width) / float(model.layers[i + 1].strides[1])))
            assert out_width == list(hls_model.get_layers())[i + 3].attributes['out_width']
            if in_width % model.layers[i + 1].strides[1] == 0:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - model.layers[i + 1].strides[1], 0)
            else:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - (in_width % model.layers[i + 1].strides[1]), 0)
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            assert pad_left == list(hls_model.get_layers())[i + 3].attributes['pad_left']
            assert pad_right == list(hls_model.get_layers())[i + 3].attributes['pad_right']

          elif list(hls_model.get_layers())[i + 3].attributes['padding'] == 'valid':
            if list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_first':
              in_height = model.layers[i + 1].input_shape[2]
              in_width = model.layers[i + 1].input_shape[3]
            elif list(hls_model.get_layers())[i + 3].attributes['data_format'] == 'channels_last':
              in_height = model.layers[i + 1].input_shape[1]
              in_width = model.layers[i + 1].input_shape[2]

            out_width = int(math.ceil(float(in_width - model.layers[i + 1].pool_size[0] + 1) / float(model.layers[i + 1].strides[1])))
            out_height = int(math.ceil(float(in_height - model.layers[i + 1].pool_size[1] + 1) / float(model.layers[i + 1].strides[0])))
        
            assert list(hls_model.get_layers())[i + 3].attributes['out_height'] == out_height
            assert list(hls_model.get_layers())[i + 3].attributes['out_width'] == out_width
            assert list(hls_model.get_layers())[i + 3].attributes['pad_top'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_bottom'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_left'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_right'] == 0
      
    elif poolings == 'MaxPooling1D' or poolings == 'AveragePooling2D':
        input_shape = (10, 128, 4)
        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', 
                            input_shape=input_shape[1:], kernel_initializer='normal', use_bias=False,
                            data_format='channels_last'))
        model.add(MaxPooling1D(pool_size=2, strides=None, padding=padds, data_format=chans))
        model.add(AveragePooling1D(pool_size=2, strides=None, padding=padds, data_format=chans))
        
        hls_model = hls4ml.converters.convert_from_keras_model(model)

        for i in range(2):
          assert list(hls_model.get_layers())[i + 3].attributes['name'] == model.layers[i + 1]._name
          assert list(hls_model.get_layers())[i + 3].attributes['class_name'][-2] == str(1) 
          assert list(hls_model.get_layers())[i + 3].attributes['n_in'] == model.layers[i + 1].input_shape[1]
          assert list(hls_model.get_layers())[i + 3].attributes['n_filt'] == model.layers[i + 1].input_shape[2]
          assert list(hls_model.get_layers())[i + 3].attributes['pool_size'] == model.layers[i + 1].pool_size[0]
          assert list(hls_model.get_layers())[i + 3].attributes['stride'] == model.layers[i + 1].strides[0]
          assert list(hls_model.get_layers())[i + 3].attributes['padding'] == model.layers[i + 1].padding
          
          out_same	= math.ceil(float(model.layers[i + 1].input_shape[1]) / float(model.layers[i + 1].strides[0]))
          out_valid	= math.ceil(float(model.layers[i + 1].input_shape[1] - model.layers[i + 1].pool_size[0] + 1) / model.layers[i + 1].strides[0])

          if list(hls_model.get_layers())[i + 3].attributes['padding'] == 'same':
            assert list(hls_model.get_layers())[i + 3].attributes['n_out'] == out_same 
            if model.layers[i + 1].input_shape[1] % model.layers[i + 1].strides[0] == 0:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - model.layers[i + 1].strides[0], 0)
            else:
              pad_along_width = max(model.layers[i + 1].pool_size[0] - (model.layers[i + 1].input_shape[1] % model.layers[i + 1].strides[0]), 0)
            assert list(hls_model.get_layers())[i + 3].attributes['pad_left'] == pad_along_width // 2
            assert list(hls_model.get_layers())[i + 3].attributes['pad_right'] == pad_along_width - pad_along_width // 2

          elif list(hls_model.get_layers())[i + 3].attributes['padding'] == 'valid':
            assert list(hls_model.get_layers())[i + 3].attributes['n_out'] == out_valid
            assert list(hls_model.get_layers())[i + 3].attributes['pad_left'] == 0
            assert list(hls_model.get_layers())[i + 3].attributes['pad_right'] == 0




merge_layers = [Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate]
# @pytest.mark.parametrize('merges', merge_layers)
# def test_merge():
#   input1 = tf.keras.layers.Input(shape=(16,))
#   x1 = tf.keras.layers.Dense(8, activation='relu')(input1)

#   input2 = tf.keras.layers.Input(shape=(32,))
#   x2 = tf.keras.layers.Dense(8, activation='relu')(input2)  

#   added = tf.keras.layers.Add()([x1, x2])
#   out = tf.keras.layers.Dense(4)(added)
#   model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

#   hls_model = hls4ml.converters.convert_from_keras_model(model)

  # assert list(hls_model.get_layers())[6].attributes['op'] == model.layers[4]._name

  # if model.layers[4]._name.lower() == 'concatenate':
    # rank = 

  # if model.layers[4]._name.lower() == 'concatenate':
  #   rank = model.layers[4]._name
  #   pass
  # else:
  #   list(hls_model.get_layers())[6].attributes['class_name'] = 'Merge'


#   print()
#   print(list(hls_model.get_layers())[6].attributes)
#   print()
#   print(model.layers[4]._name)

#   # 'name': 'subtract', 'class_name': 'Merge', 'inputs': ['dense_relu', 'dense_1_relu'], 'op': 'subtract'
  
#   # keras model -> Input, Input, Dense, Dense, Add, Dense
#   # hls_model   -> Input, Input, Dense, Activation, Dense, Activation, Merge, Dense

# test_merge()



# merge_layers = [Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate]
# # @pytest.mark.parametrize("merges", merge_layers)
# def test_merge():
#     model1 = tf.keras.models.Sequential()
#     model1.add(Dense(1, input_shape=(2,), activation='sigmoid'))

#     model2 = tf.keras.models.Sequential()
#     model2.add(Dense(1, input_shape=(1,), activation='sigmoid'))

#     model = tf.keras.models.Sequential()
#     model.add(Concatenate([model1, model2]))
#     model.add(Activation(activation='relu'))

#     hls_model = hls4ml.converters.convert_from_keras_model(model)

#     # print(model.layers)
#     # print()
#     # print(m.__dict__.keys())

#     # input_shape = (2, 3, 4)
#     # x1 = tf.random.normal(input_shape)
#     # print(x1)
#     # x2 = tf.random.normal(input_shape)
#     # print(x2)
#     # y = tf.keras.layers.Add()([x1, x2])
#     # print(y)



# BatchNormalization layer dissappears when keras model is converted to hls model
# keras_normalization_function = [BatchNormalization]
# @pytest.mark.parametrize("normalization_function", keras_normalization_function)
# def test_batch_normalization():
#   model = tf.keras.models.Sequential()
#   model.add(Dense(64, 
#               input_shape=(16,), 
#               name='Dense', 
#               use_bias=True,
#               kernel_initializer='lecun_uniform',
#               bias_initializer='zeros'))
#   model.add(BatchNormalization())
#   model.add(Activation(activation='elu', name='Activation'))

#   hls_model = hls4ml.converters.convert_from_keras_model(model)
#   print(hls_model.get_layers())
#   print()
#   print(model.layers)

# test_batch_normalization()



# keras_dense_layers = [BinaryDense, TernaryDense]
# @pytest.mark.parametrize("dense_layers", keras_dense_layers)
# def test_dense_layer():
    # model = tf.keras.models.Sequential()


# test_dense_layer()