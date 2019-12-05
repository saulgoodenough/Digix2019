#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
from keras import regularizers
from keras.layers import Input, Dense, concatenate, BatchNormalization, Dropout,Embedding, Reshape, LSTM, Conv1D, GlobalAveragePooling1D
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy import sparse


def acc(y_true, y_pred):
    y_pred_categ = K.round(y_pred)
    accuracy = K.classification_error(y_true, y_pred_categ)
    return accuracy

def fcnmodel():
    regularizer_para_1 = regularizers.l1(0.01)
    regularizer_para_2 = regularizers.l2(0.01)
    learning_rate = 0.01
    dim_bhv = 8
    dim_bsc1 = 7
    dim_bsc2 = 739  # 'city prodName color ct carrier'
    dim_act = 9401
    dim_app = 56340
    dim_app1 = 40
    dropout_rate = 0.01
    #加一个卷积试一下
    input_app = Input(shape=(dim_app,), name='in_app')  # , dtype = 'int32') , sparse=True
    input_app_1 = Reshape((dim_app, 1), input_shape=(dim_app,))(input_app)
    encode_app = Conv1D(256, 10000, activation='relu')(input_app_1)
    encode_app = GlobalAveragePooling1D()(encode_app)
    encode_app = BatchNormalization()(encode_app)
    encode_app = Dropout(dropout_rate)(encode_app)
    encode_app = Dense(128, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(encode_app)
    encode_app = BatchNormalization()(encode_app)
    encode_app = Dropout(dropout_rate)(encode_app)
    encode_app = Dense(64, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',
                       bias_initializer='zeros')(encode_app)
    encode_app = BatchNormalization()(encode_app)


    input_bhv = Input(shape=(dim_bhv,), name='in_bhv')
    encode_bhv = Dense(128, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(input_bhv)
    encode_bhv = BatchNormalization()(encode_bhv)
    encode_bhv = Dropout(dropout_rate)(encode_bhv)
    encode_bhv = Dense(64, activation='relu',  kernel_regularizer=regularizer_para_1, kernel_initializer='random_normal',bias_initializer='zeros')(encode_bhv)
    encode_bhv = BatchNormalization()(encode_bhv)
    encode_bhv = Dense(32, activation='relu',  kernel_regularizer=regularizer_para_1,
                       kernel_initializer='random_normal', bias_initializer='zeros')(encode_bhv)
    encode_bhv = BatchNormalization()(encode_bhv)


    input_bsc1 = Input(shape=(dim_bsc1,), name='in_bsc1')
    encode_bsc1 = Dense(64, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(input_bsc1)
    encode_bsc1 = BatchNormalization()(encode_bsc1)
    encode_bsc1 = Dropout(dropout_rate)(encode_bsc1)
    encode_bsc1 = Dense(32, activation='relu',  kernel_regularizer=regularizer_para_2,kernel_initializer='random_normal',bias_initializer='zeros')(encode_bsc1)
    encode_bsc1 = BatchNormalization()(encode_bsc1)



    input_app1 = Input(shape=(dim_app1,), name='input_app1')
    encode_app2 = Dense(64, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(input_app1)
    encode_app2 = BatchNormalization()(encode_app2)
    encode_app2 = Dropout(dropout_rate)(encode_app2)
    encode_app2 = Dense(64, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(encode_app2)
    encode_app2 = BatchNormalization()(encode_app2)




    input_bsc2 = Input(shape=(dim_bsc2,), name='in_bsc2')  # , dtype = 'int32'), sparse=True
    encode_bsc2 = Dense(128, activation='relu', kernel_regularizer= regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(input_bsc2)
    encode_bsc2 = BatchNormalization()(encode_bsc2)
    encode_bsc2 = Dropout(dropout_rate)(encode_bsc2)
    encode_bsc2 = Dense(64, activation='relu', name='encode_bsc2', kernel_regularizer=regularizer_para_2,
                        kernel_initializer='random_normal', bias_initializer='zeros')(encode_bsc2)
    encode_bsc2 = BatchNormalization()(encode_bsc2)
    encode_bsc2 = Dropout(dropout_rate)(encode_bsc2)
    encode_bsc2 = Dense(32, activation='relu', kernel_regularizer= regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(encode_bsc2)
    encode_bsc2 = BatchNormalization()(encode_bsc2)

    #加入卷积试一下看一下效果

    input_act = Input(shape=(dim_act,), name='in_act')  # , dtype = 'int32') , sparse=True
    input_act_1 = Reshape((dim_act, 1), input_shape=(dim_act,))(input_act)
    encode_act = Conv1D(256, 1, activation='relu')(input_act_1)
    encode_act = GlobalAveragePooling1D()(encode_act)
    encode_act = BatchNormalization()(encode_act)
    encode_act = Dropout(dropout_rate)(encode_act)
    encode_act = Dense(128, activation='relu', kernel_regularizer=regularizer_para_1, kernel_initializer='random_normal',bias_initializer='zeros')(encode_act)
    encode_act = BatchNormalization()(encode_act)
    encode_act = Dropout(dropout_rate)(encode_act)
    encode_act = Dense(64, activation="relu", kernel_regularizer=regularizer_para_1, kernel_initializer='random_normal',bias_initializer='zeros')(encode_act)
    encode_act = BatchNormalization()(encode_act)
    


    # --- merge ---
    merge = concatenate([encode_bhv, encode_bsc1,  encode_bsc2, encode_act, encode_app2, encode_app])
    merge = Dense(256, activation = 'relu')(merge)
    merge = BatchNormalization()(merge)
    merge = Dropout(dropout_rate)(merge)
    merge = Dense(128, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal', bias_initializer='zeros')(merge)
    merge = BatchNormalization()(merge)
    merge = Dropout(dropout_rate)(merge)
    merge = Dense(64, activation='relu', kernel_regularizer= regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(merge)
    # output
    out_regress = Dense(1, activation='relu',  name='out_regress')(merge)
    out_clf = Dense(6, activation='softmax', name='out_clf')(merge)

    model = Model(inputs=[input_bhv, input_bsc1, input_bsc2, input_act, input_app1, input_app],
                  outputs=[out_regress, out_clf])

    model.summary()
    optimizer = optimizers.Adam() # lr=learning_rate
    #optimizer = optimizers.RMSprop() #lr=learning_rate

    model.compile(optimizer=optimizer, loss={'out_regress':'mean_squared_error','out_clf': 'categorical_crossentropy'}, loss_weights=[1, 10000000000], metrics={'out_regress': 'mae','out_clf': 'acc'})
    return model


if __name__ == '__main__':

    X_app_6 = sparse.load_npz('train_sparse_matrix.npz')


    X_bhv = np.load('behav_train.npy')
    X_bhv_log = np.load('log_behav_train.npy')

    X_bsc1 = np.load('basic1_train.npy')
    X_bsc2 = np.load('basic2_train.npy').item()
    X_act = np.load('act1hot_train.npy').item()

    X_app_1 = pd.read_csv("data_train_appclass.csv", sep =",")
    X_app_2 = X_app_1.iloc[:, -40:].values
    X_app = np.array(X_app_2).astype(int)

    print('X_bhv_log', type(X_bhv_log), X_bhv_log.shape)
    print('X_bsc1', type(X_bsc1), X_bsc1.shape)
    print('X_bsc2', type(X_bsc2), X_bsc2.shape)
    print('X_act', type(X_act), X_act.shape)
    print('X_app', type(X_app), X_app.shape)
    print('X_app_6', type(X_app_6), X_app_6.shape)


    Y = np.load('age.npy')
    Y_1hot = to_categorical(Y - 1, 6)

    batch_sz = 128
    valid_split = 0.05
    epc = 10

    fname_model = 'model/dnn'
    checkpoint = ModelCheckpoint(fname_model, monitor='val_out_clf_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_out_clf_acc', min_delta=0, patience=0, verbose=1, mode='max')
    model = fcnmodel()
    model.fit(x=[X_bhv_log, X_bsc1, X_bsc2, X_act, X_app, X_app_6], y=[Y, Y_1hot],
              batch_size=batch_sz, epochs=epc, validation_split=valid_split,
              callbacks=[checkpoint, early_stopping], verbose=1)  # , class_weight = )
    #model.save(fname_model)
