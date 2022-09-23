from venv import create
import tensorflow as tf
import keras
from keras import layers
from keras.initializers import GlorotNormal
from keras.regularizers import l2

def qNetFC(input_shape, output_shape):
    
    inputs = layers.Input(shape=input_shape)
    
    # Hidden layers
    lyr1 = layers.Dense(32, activation='relu')(inputs)
    lyr2 = layers.Dense(64, activation='relu')(lyr1)
    lyr3 = layers.Dense(32, activation='relu')(lyr2)
    
    #Output
    action = layers.Dense(output_shape, activation='linear')(lyr3)
    
    return keras.Model(inputs=inputs, outputs=action)

def DDPGActor(input_shape, output_shape):
    
    return qNetFC(input_shape=input_shape, output_shape=output_shape)

def DDPGCritic(input_obs, input_action):

    input_obs = layers.Input(shape=input_obs)
    input_action = [layers.Input(2) for i in range(input_action)]
    action = layers.Concatenate()(input_action)
    inputs = [input_obs, action]
    cat = layers.Concatenate(axis=-1)(inputs)

    # hidden layer 1
    h1_ = layers.Dense(300, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(cat)
    h1_b = layers.BatchNormalization()(h1_)
    h1 = layers.Activation('relu')(h1_b)
    
    # hidden_layer 2
    h2_ = layers.Dense(400, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h1)
    h2_b = layers.BatchNormalization()(h2_)
    h2 = layers.Activation('relu')(h2_b)
    # output layer(actions)
    output_ = layers.Dense(1, kernel_initializer=GlorotNormal(), kernel_regularizer=l2(0.01))(h2)
    output_b = layers.BatchNormalization()(output_)
    output = layers.Activation('linear')(output_b)
    return keras.Model(inputs,output) 

if __name__ == '__main__':
    
    model = qNetFC((10, 10, 3), 2)
    #print(type(model))
    