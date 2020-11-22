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
                                    AveragePooling2D, Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate
import math 
from tensorflow.keras import backend as K 
from numpy.testing import  assert_allclose                    

'''
There are four functions for all layers: 
1.Making Keras model
2.Converting Keras model to HLS one 
3.Testing Conversion process
4.Predicting and comparing results from both models 
'''
# Dense Layer
def make_dense_model():
  '''
  This function makes a Sequential() model with 2 layers: Dense and Activation. Afterwards 
  it is compiled with Adam optimizer and return Keras model. 
  '''
  model = tf.keras.models.Sequential()
  model.add(Dense(2, 
            input_shape=(1,), 
            name='Dense', 
            use_bias=True,
            kernel_initializer= tf.keras.initializers.RandomUniform(minval=1, maxval=10),
            bias_initializer='zeros', 
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None))
  model.add(Activation(activation='elu', name='Activation'))
  model.compile(optimizer='adam', loss='mse')
  return model

def convert_dense_model():
  '''
  The Keras model is gotten by make_dense_model() function and assigned to 
  'model' variable. Then the Keras model is converted to the HLS model by 
  means of HLS configuration and return it. 
  '''
  model = make_dense_model()
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model

def test_dense_conversion():
  '''
  The Keras and HLS model are gotten by make_dense_model() and convert_dense_model()
  functions, and assigned to 'model' and 'hls_model' variables. Lastly, attributes of
  both models are compared by Pytest's assert statement.
  '''
  model = make_dense_model()
  hls_model = convert_dense_model()
  assert len(model.layers) + 1 == len(hls_model.get_layers())
  assert list(hls_model.get_layers())[0].attributes['class_name'] == "InputLayer"
  assert list(hls_model.get_layers())[1].attributes["class_name"] == model.layers[0]._name
  assert list(hls_model.get_layers())[2].attributes['class_name'] == model.layers[1]._name
  assert list(hls_model.get_layers())[0].attributes['input_shape'] == list(model.layers[0].input_shape[1:])
  assert list(hls_model.get_layers())[1].attributes['n_in'] == model.layers[0].input_shape[1:][0]
  assert list(hls_model.get_layers())[1].attributes['n_out'] == model.layers[0].output_shape[1:][0]
  assert list(hls_model.get_layers())[2].attributes['activation'] == str(model.layers[1].activation).split()[1]
  assert list(hls_model.get_layers())[1].attributes['activation'] == str(model.layers[0].activation).split()[1]

def test_dense_prediction():
  '''
  The Keras and HLS model are gotten by make_dense_model() and convert_dense_model()
  functions, and assigned to 'model' and 'hls_model' variables. X_input is generated 
  by Numpy's random generator and and given to the Keras and HLS model. After getting
  prediction results according to the both models we compare them. 
  '''
  model = make_dense_model()
  hls_model = convert_dense_model()
  X_input = np.random.rand(1,)
  keras_prediction = model.predict(X_input)
  hls_model.compile()
  hls_prediction = hls_model.predict(X_input)
  assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction)))) < 3


# LeakyReLU and ELU Activation Layers 
'''
keras_activation_functions list keep two types of activation function and 
@pytest.mark.parametrize allows that we can create the Keras model with 
two of them.
'''
keras_activation_functions = [LeakyReLU, ELU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def make_leakyrelu_elu(activation_functions):
  '''
  This function makes a Sequential() model with 2 layers: Dense and Activation. Afterwards 
  it is compiled with Adam optimizer and return Keras model. 
  '''
  model = tf.keras.models.Sequential()
  model.add(Dense(64, 
            input_shape=(1,), 
            name='Dense', 
            kernel_initializer='lecun_uniform', 
            kernel_regularizer=None))
  model.add(activation_functions(alpha=1.0))
  model.compile(optimizer='adam', loss='mse')
  return model
 
def convert_activation_leakyrelu_elu(activation_functions):
  '''
  The Keras model is gotten by make_leakyrelu_elu() function and assigned to 
  'model' variable. Then the Keras model is converted to the HLS model by 
  means of HLS configuration and return it. 
  '''
  model = make_leakyrelu_elu(activation_functions)
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model

@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_leakyrelu_elu_conversion(activation_functions):
  '''
  The Keras and HLS model are gotten by make_leakyrelu_elu() and 
  convert_activation_leakyrelu_elu() functions, and assigned to 'model' and 
  'hls_model' variables. Lastly, attributes of both models are compared by 
  Pytest's assert statement. Here @pytest.mark.parametrize helps us to compare 
  both functions at the same time.  
  '''
  model = make_leakyrelu_elu(activation_functions)
  hls_model = convert_activation_leakyrelu_elu(activation_functions)
  assert len(model.layers) + 1 == len(hls_model.get_layers())
  if activation_functions == 'ELU':
    assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ELU'
  elif activation_functions == 'LeakyReLU':
    assert list(hls_model.get_layers())[2].attributes['class_name'] == 'LeakyReLU'

@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_activation_leakyrelu_elu_conversion_prediction(activation_functions):
  '''
  The Keras and HLS model are gotten by make_leakyrelu_elu() and 
  convert_activation_leakyrelu_elu() functions, and assigned to 'model' and 
  'hls_model' variables. X_input is generated by Numpy's random generator and 
  given to the Keras and HLS model. After getting prediction results according 
  to the both models we compare them. 
  '''
  model = make_leakyrelu_elu(activation_functions)
  hls_model = convert_activation_leakyrelu_elu(activation_functions)
  X_input = np.random.rand(1)
  keras_prediction = model.predict(X_input)
  hls_model.compile()
  hls_prediction = hls_model.predict(X_input)
  assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction)))) < 3

  
# ThresholdedReLU Activation Layer
keras_activation_functions = [ThresholdedReLU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def make_thresholdedrelu(activation_functions):
  '''
  keras_activation_functions list keeps only one type of activation function 
  -> ThresholdedReLU and @pytest.mark.parametrize allows that we can create 
  the Keras model with it.
  '''
  model = tf.keras.models.Sequential()
  model.add(Dense(64, 
            input_shape=(1,), 
            name='Dense', 
            kernel_initializer='lecun_uniform', 
            kernel_regularizer=None))
  model.add(activation_functions(theta=1.0))
  model.compile(optimizer='adam', loss='mse')
  return model

def convert_thresholdedrelu_model(activation_functions):
  '''
  The Keras model is gotten by make_thresholdedrelu() function and assigned to 
  'model' variable. Then the Keras model is converted to the HLS model by 
  means of HLS configuration and return it. 
  '''
  model = make_thresholdedrelu(activation_functions)
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model

@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_thresholdedrelu_conversion(activation_functions):
  '''
  The Keras and HLS model are gotten by make_thresholdedrelu() and 
  convert_thresholdedrelu_model() functions, and assigned to 'model' and 
  'hls_model' variables. Lastly, attributes model are compared by Pytest's 
  assert statement.  
  '''
  model = make_thresholdedrelu(activation_functions)
  hls_model = convert_thresholdedrelu_model(activation_functions)
  assert len(model.layers) + 1 == len(hls_model.get_layers())
  if activation_functions == 'ThresholdedReLU':
    assert list(hls_model.get_layers())[2].attributes['class_name'] == 'ThresholdedReLU'

'''
  The Keras and HLS model are gotten by make_thresholdedrelu() and 
  convert_thresholdedrelu_model() functions, and assigned to 'model' and
  'hls_model' variables. X_input is generated by Numpy's random generator and 
  given to the Keras and HLS model. After getting prediction results according 
  to the both models we compare them. 
  '''
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_thresholdedrelu_prediction(activation_functions):
  model = make_thresholdedrelu(activation_functions)
  hls_model = convert_thresholdedrelu_model(activation_functions)
  X_input = np.random.rand(1,)
  keras_prediction = model.predict(X_input)
  hls_model.compile()
  hls_prediction = hls_model.predict(X_input)
  assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction)))) < 3 


'''
Everything is same as in ThresholdedReLU Activation Layer above except making 
model with PReLU Activation Layer.
'''
# PReLU Activation Layer
keras_activation_functions = [PReLU]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def make_prelu_model(activation_functions):
  model = tf.keras.models.Sequential()
  model.add(Dense(64, 
            input_shape=(1,), 
            name='Dense', 
            kernel_initializer='lecun_uniform', 
            kernel_regularizer=None))
  model.add(activation_functions(alpha_initializer="zeros",))
  model.compile(optimizer='adam', loss='mse')
  return model

def convert_prelu_model(activation_functions):
  model = make_prelu_model(activation_functions)
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model

@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_prelu_conversion(activation_functions):
  model = make_prelu_model(activation_functions)
  hls_model = convert_prelu_model(activation_functions)
  assert len(model.layers) + 1 == len(hls_model.get_layers())
  if activation_functions == 'PReLU':
    assert list(hls_model.get_layers())[2].attributes['class_name'] == 'PReLU'

@pytest.mark.parametrize("activation_functions", keras_activation_functions)
def test_prelu_prediction(activation_functions):
  model = make_prelu_model(activation_functions)
  hls_model = convert_prelu_model(activation_functions)
  X_input = np.random.rand(1,)
  keras_prediction = model.predict(X_input)
  hls_model.compile()
  hls_prediction = hls_model.predict(X_input)
  assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction)))) < 3


# Activation layer 
keras_activation_functions = [Activation]
@pytest.mark.parametrize("activation_functions", keras_activation_functions)    
def make_activation_model(activation_functions):
  '''
  Everything is same as in ThresholdedReLU Activation Layer above except making 
  model with simple Activation Layer.
  '''
  model = tf.keras.models.Sequential()
  model.add(Dense(64, 
            input_shape=(1,), 
            name='Dense', 
            kernel_initializer='lecun_uniform', 
            kernel_regularizer=None))
  model.add(activation_functions(activation='relu', name='Activation'))
  model.compile(optimizer='adam', loss='mse')
  return model

def convert_activation_model(activation_functions):
  model = make_activation_model(activation_functions)
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model

@pytest.mark.parametrize("activation_functions", keras_activation_functions)  
def test_activation_conversion(activation_functions):
  model = make_activation_model(activation_functions)
  hls_model = convert_activation_model(activation_functions)
  assert len(model.layers) + 1 == len(hls_model.get_layers())
  if activation_functions == 'Activation':
    assert list(hls_model.get_layers())[2].attributes["activation"] == str(model.layers[1].activation).split()[1] 

@pytest.mark.parametrize("activation_functions", keras_activation_functions)  
def test_activation_prediction(activation_functions):
  model = make_activation_model(activation_functions)
  hls_model = convert_activation_model(activation_functions)
  X_input = np.random.rand(1,)
  keras_prediction = model.predict(X_input)
  hls_model.compile()
  hls_prediction = hls_model.predict(X_input)
  assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction)))) < 3
  

# Conv1D Layer
keras_conv1d = [Conv1D]
padds_options = ['same', 'valid']
@pytest.mark.parametrize("conv1d", keras_conv1d)
@pytest.mark.parametrize("padds", padds_options)
def make_conv1d_model(conv1d, padds):
  model = tf.keras.models.Sequential()
  input_shape = (10, 128, 4)
  model.add(conv1d(filters=32, 
                  kernel_size=3, 
                  strides=2, 
                  padding=padds, 
                  activation='relu', 
                  input_shape=input_shape[1:], 
                  kernel_initializer='normal', 
                  use_bias=False,
                  data_format='channels_last'))
  model.add(Activation(activation='relu'))
  model.compile(optimizer='adam', loss='mse')
  return model

def convert_conv1d_model(conv1d, padds):
  '''
  The Keras model is gotten by make_conv1d_model() function and assigned to 
  'model' variable. Then the Keras model is converted to the HLS model by 
  means of HLS configuration and return it. 
  '''
  model = make_conv1d_model(conv1d, padds)
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model

@pytest.mark.parametrize("conv1d", keras_conv1d)
@pytest.mark.parametrize("padds", padds_options)
def test_conv1d_conversion(conv1d, padds):
  model = make_conv1d_model(conv1d, padds)
  hls_model = convert_conv1d_model(conv1d, padds)
  assert len(model.layers) + 2 == len(hls_model.get_layers()) 
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
  
@pytest.mark.parametrize("conv1d", keras_conv1d)
@pytest.mark.parametrize("padds", padds_options)
def test_conv1d_prediction(conv1d, padds):
  model = make_conv1d_model(conv1d, padds)
  hls_model = convert_conv1d_model(conv1d, padds)
  X_input = np.random.rand(10, 128,4)
  keras_prediction = model.predict(X_input)
  hls_model.compile()
  hls_prediction = hls_model.predict(X_input)
  if padds_options == 'same':
    assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction.reshape(10,64,32))))) < 3
  elif padds_options == 'valid':
    assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction.reshape(10,63,32))))) < 3


# MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D Layers 
pooling_layers = [MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D]
padds_options = ['same', 'valid']
chans_options = ['channels_first', 'channels_last']
@pytest.mark.parametrize("poolings", pooling_layers)
@pytest.mark.parametrize("padds", padds_options)
@pytest.mark.parametrize("chans", chans_options)
def test_pooling(poolings, padds, chans):
    model = tf.keras.models.Sequential()
    if poolings == 'MaxPooling2D' or poolings == 'AveragePooling2D':
      def make_pooling_model():
        model.add(Conv2D(1, (3,3), activation='relu', input_shape=(8, 8, 1)))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        model.compile(optimizer='adam', loss='mse')
        return model
      
      def convert_pooling_model():
        '''
        The Keras model is gotten by make_pooling_model() function and assigned to 
        'model' variable. Then the Keras model is converted to the HLS model by 
        means of HLS configuration and return it. 
        '''
        model = make_pooling_model()
        config = hls4ml.utils.config_from_keras_model(model)
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
        return hls_model

      def test_pooling_conversion():
        model = make_pooling_model()
        hls_model = convert_pooling_model()
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
            
            # Width
            in_width = model.layers[i + 1].input_shape[2]
            if model.layers[i + 1].data_format == 'channels_first':
              in_height = model.layers[1].input_shape[i + 3]
            out_width = int(math.ceil(float(in_width) / float(model.layers[i + 1].strides[1])))
            assert out_width == list(hls_model.get_layers())[i + 3].attributes['out_width']

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

      def test_pooling_prediction():
        X_input = np.random.rand(10, 128,4)
        keras_prediction = model.predict(X_input)
        config = hls4ml.utils.config_from_keras_model(model)
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
        hls_model.compile()
        hls_prediction = hls_model.predict(X_input)
        assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction)))) < 3

      
    elif poolings == 'MaxPooling1D' or poolings == 'AveragePooling1D':
      def make_pooling_model():
        input_shape = (10, 128, 4)
        model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', 
                            input_shape=input_shape[1:], kernel_initializer='normal', use_bias=False,
                            data_format='channels_last'))
        model.add(MaxPooling1D(pool_size=2, strides=None, padding=padds, data_format=chans))
        model.add(AveragePooling1D(pool_size=2, strides=None, padding=padds, data_format=chans))
        model.compile(optimizer='adam', loss='mse')
        return model 

      def convert_pooling_model():
        '''
        The Keras model is gotten by make_pooling_model() function and assigned to 
        'model' variable. Then the Keras model is converted to the HLS model by 
        means of HLS configuration and return it. 
        '''
        model = make_pooling_model()
        config = hls4ml.utils.config_from_keras_model(model)
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
        return hls_model

      def test_pooling_conversion():
        model = make_pooling_model()
        hls_model = convert_pooling_model()
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

          elif list(hls_model.get_layers())[i + 3].attributes['padding'] == 'valid':
            assert list(hls_model.get_layers())[i + 3].attributes['n_out'] == out_valid

      def test_pooling_prediction():
        X_input = np.random.rand(10, 128,4)
        keras_prediction = model.predict(X_input)
        config = hls4ml.utils.config_from_keras_model(model)
        hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
        hls_model.compile()
        hls_prediction = hls_model.predict(X_input)
        assert round(np.average(np.subtract(np.abs(keras_prediction), np.abs(hls_prediction)))) < 3
      

# Reshape layer
def make_reshape_model():
  model = tf.keras.models.Sequential()
  model.add(Dense(12, 
            input_shape=(1,), 
            name='Dense', 
            use_bias=True,
            kernel_initializer= tf.keras.initializers.RandomUniform(minval=1, maxval=10),
            bias_initializer='zeros', 
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None))
  model.add(Reshape((3,4)))
  model.add(Activation(activation="elu", name='Activation'))
  model.compile(optimizer='adam', loss='mse')
  return model

def convert_reshape_model():
  '''
  The Keras model is gotten by make_reshape_model() function and assigned to 
  model' variable. Then the Keras model is converted to the HLS model by 
  means of HLS configuration and return it. 
  '''
  model = make_reshape_model()
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model

def test_reshape_conversion():
  model = make_reshape_model()
  hls_model = convert_reshape_model()
  assert len(model.layers) + 1 == len(hls_model.get_layers())
  assert list(hls_model.get_layers())[2].attributes['target_shape'] == list(model.layers[1].target_shape)


# Conv2D Layer
keras_conv2d = [Conv2D]
padds_options = ['same', 'valid']
chans_options = ['channels_first', 'channels_last']
@pytest.mark.parametrize("conv2d", keras_conv2d)
@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
def make_conv2d_model(conv2d, chans, padds):
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
  model.compile(optimizer='adam', loss='mse')
  return model

def convert_conv2d_model(conv2d, chans, padds):
  '''
  The Keras model is gotten by make_conv2d_model() function and assigned to 
  model' variable. Then the Keras model is converted to the HLS model by 
  means of HLS configuration and return it. 
  '''
  model = make_conv2d_model(conv2d, chans, padds)
  config = hls4ml.utils.config_from_keras_model(model)
  hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config)
  return hls_model 

@pytest.mark.parametrize("conv2d", keras_conv2d)
@pytest.mark.parametrize("chans", chans_options)
@pytest.mark.parametrize("padds", padds_options)
def test_conv2d_conversion(conv2d, chans, padds):
  model = make_conv2d_model(conv2d, chans, padds)
  hls_model = convert_conv2d_model(conv2d, chans, padds)
  assert len(model.layers) + 2 == len(hls_model.get_layers()) 

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


merge_layers = [Add, Subtract, Multiply, Average, Maximum, Minimum, Concatenate]
@pytest.mark.parametrize('merges', merge_layers)
def test_merge(merges):
  input1 = tf.keras.layers.Input(shape=(16,))
  x1 = tf.keras.layers.Dense(8, activation='relu')(input1)

  input2 = tf.keras.layers.Input(shape=(32,))
  x2 = tf.keras.layers.Dense(8, activation='relu')(input2)  

  added = merges()([x1, x2])
  out = tf.keras.layers.Dense(4)(added)
  model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

  hls_model = hls4ml.converters.convert_from_keras_model(model)

  if model.layers[4]._name == 'concatenate':
    assert list(hls_model.get_layers())[6].attributes['axis'] == model.layers[4].axis
    assert list(hls_model.get_layers())[6].attributes["class_name"].lower() == model.layers[4]._name

  else:
    assert list(hls_model.get_layers())[6].attributes['class_name'] == 'Merge'

  assert len(list(hls_model.get_layers())[6].attributes['inputs']) <= 2