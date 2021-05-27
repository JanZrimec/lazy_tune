
# coding: utf-8

# In[ ]:

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten, Concatenate
from tensorflow. keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import sys  
sys.path.insert(0, '../')
from my_utils import coef_det_k

def load_data(fname):
    # X is multi-variable array
    # Y contains single variable - fix shape for Keras

    npzfile = np.load(fname)
    Xh_train = npzfile['arr_0']
    Xh_test = npzfile['arr_1']
    Xv_train = npzfile['arr_2']
    Xv_test = npzfile['arr_3']
    Y_train = npzfile['arr_4']
    Y_test = npzfile['arr_5']

    X_train = list()
    X_train.append(Xh_train)
    X_train.append(Xv_train)
    X_test = list()
    X_test.append(Xh_test)
    X_test.append(Xv_test)

    Y_train = Y_train.astype(np.float32).reshape((-1,1))
    Y_test = Y_test.astype(np.float32).reshape((-1,1))

    return X_train, X_test, Y_train, Y_test

def Params():
    params = {
        'kernel_size1': [10, 20, 30, 40],
        'filters1': [32, 64, 128],
        'dilation1': [1, 2, 4],
        'pool_size1': [1, 2, 4],
        'stride1': [1, 2],
        'dropout1': (0, 1),
        'kernel_size2': [10, 20, 30, 40],
        'filters2': [32, 64, 128],
        'dilation2': [1, 2, 4],
        'pool_size2': [1, 2, 4],
        'stride2': [1, 2],
        'dropout2': (0, 1),
        'kernel_size3': [10, 20, 30, 40],
        'filters3': [32, 64, 128],
        'dilation3': [1, 2, 4],
        'pool_size3': [1, 2, 4],
        'stride3': [1, 2],
        'dropout3': (0, 1),
        'dense5': [32, 64, 128],
        'dropout5': (0, 1),
        'dense6': [32, 64, 128],
        'dropout6': (0, 1)
    }
    return {k: hp.choice(k, v) if type(v) == list else hp.uniform(k, v[0], v[1]) for k, v in params.items()}

def POC_model(shapes, p):

    X_input1 = Input(shape = shapes[0])
    X_input2 = Input(shape = shapes[1])

    X = Conv1D(filters=int(p['filters1']),kernel_size=int(p['kernel_size1']),strides=1,dilation_rate=int(p['dilation1']),activation='relu',kernel_initializer='he_uniform')(X_input1)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout1']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size1']), strides=int(p['stride1']), padding='same')(X)

    X = Conv1D(filters=int(p['filters2']),kernel_size=int(p['kernel_size2']),strides=1,dilation_rate=int(p['dilation2']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout2']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size2']), strides=int(p['stride2']), padding='same')(X)
    
    X = Conv1D(filters=int(p['filters3']),kernel_size=int(p['kernel_size3']),strides=1,dilation_rate=int(p['dilation3']),padding='same',activation='relu',kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout3']))(X)
    X = MaxPooling1D(pool_size=int(p['pool_size3']), strides=int(p['stride3']), padding='same')(X)

    X = Flatten()(X)
    X = Concatenate(axis=1)([X,X_input2])
    
    X = Dense(int(p['dense5']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout5']))(X)
    
    X = Dense(int(p['dense6']), activation='relu', kernel_initializer='he_uniform')(X)
    X = BatchNormalization()(X)
    X = Dropout(float(p['dropout6']))(X)
    
    X = Dense(1)(X)

    model = Model(inputs = [X_input1, X_input2], outputs = X)

    return model

def build_model(hp, input_shape):
    
    input1 = Input(input_shape[0])
    input2 = Input(input_shape[1])

    for i in range(3): #range(hp.Int('num_conv1d_layers', min_value=2, max_value=4)):
        x = Conv1D(filters=hp.Choice(f'filters_{i}', values=[32,64,128,256], default=64),
                          kernel_size=hp.Int(f'kernel_size_{i}', min_value=10, max_value=40, step=10),
                          strides=1, #hp.Choice(f'strides_{i}', values=[1,2,4], default=1), #"strides > 1 not supported in conjunction with dilation_rate > 1"
                          dilation_rate=hp.Choice(f'dilation_rate_{i}', values=[1,2,4,8], default=1),
                          activation='relu',
                          kernel_initializer='he_uniform')(input1 if i == 0 else x)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(f'conv1d_dropout_rate_{i}', min_value=0.1, max_value=0.7, sampling='linear'))(x)

    x = Flatten()(x)
    x = Concatenate(axis=1)([x,input2])

    for i in range(2):
        x = Dense(units=hp.Int(f'dense_units_{i}', min_value=64, max_value=256, step=64),
                         activation='relu',
                         kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Dropout(rate=hp.Float(f'dense_dropout_rate_{i}', min_value=0.0, max_value=0.5, sampling='linear'))(x)

    output = Dense(1)(x)

    model = Model(inputs=[input1, input2], outputs=output)
    model.compile(
        optimizer=Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='log'),
            beta_1=hp.Float('beta_1', min_value=0.5, max_value=0.95, sampling='linear'),
            beta_2=hp.Float('beta_2', min_value=0.9, max_value=0.9999, sampling='log')
        ),
        loss='mse',
        metrics=[coef_det_k]
    )
    return model
