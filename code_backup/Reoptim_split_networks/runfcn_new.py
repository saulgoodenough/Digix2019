'''
TensorflowBoys 2019
Biao Zhang
'''

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
from keras import regularizers
from keras.layers import Input, Dense, concatenate, BatchNormalization, Dropout,Embedding, Reshape, LSTM, \
    Add, Multiply, Average,Concatenate, PReLU
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy import sparse
from keras import initializers
from keras.callbacks import ReduceLROnPlateau

# initializers.glorot_normal(), keras.initializers.glorot_uniform(), keras.initializers.he_normal()
def dense_split(dim_input, dim_output_block, block_num, block_width_list = [128, 64, 32, 8]):
    last_block_dim = dim_input % block_num
    average_block_dim = dim_input // block_num
    #input_tensor = Input(shape=(dim_input,))
    input_list = []
    output_list = []
    for i in range(block_num-1):
        #input_tensor_temp = Input(shape=(average_block_dim,))
        input_tensor_temp, output_tensor_temp = dense_add(average_block_dim, block_width_list, dim_output_block)
        input_list.append(input_tensor_temp)
        output_list.append(output_tensor_temp)
    input_tensor_temp, output_tensor_temp = dense_add(last_block_dim, block_width_list, dim_output_block)
    input_list.append(input_tensor_temp)
    output_list.append(output_tensor_temp)
    input_tensor = Concatenate()(input_list)
    output_tensor = Concatenate()(output_list)
    return input_tensor, output_tensor

def dense_add_update(input_tensor, width_list, dim_output,   activation_function=None, regularizer_function = regularizers.l1(0.01),
              kernel_initializer_function = initializers.he_uniform(), bias_initializer_function = initializers.he_normal() ):
    input_list = []
    x0 = Dense(width_list[0], activation=activation_function, kernel_regularizer=regularizer_function,
                       kernel_initializer=kernel_initializer_function, bias_initializer=bias_initializer_function)(input_tensor)
    x0 = BatchNormalization()(x0)
    x0 = PReLU()(x0)
    input_list.append(x0)
    width_len = len(width_list)
    for i in range(1, width_len):
        #if width_list[i] != width_list[i-1]:
            #continue
        if i >1:
            input_temp = Concatenate()(input_list)
        else:
            input_temp = x0
        x_temp = Dense(width_list[i], activation=activation_function, kernel_regularizer=regularizer_function,
                       kernel_initializer=kernel_initializer_function, bias_initializer=bias_initializer_function)(input_temp)
        x_temp = BatchNormalization()(x_temp)
        x_temp = PReLU()(x_temp)
        input_list.append(x_temp)

    output_temp =  Concatenate()(input_list)
    output_tensor = Dense(dim_output, activation=activation_function, kernel_regularizer=regularizer_function,
                       kernel_initializer=kernel_initializer_function, bias_initializer=bias_initializer_function)(output_temp)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = PReLU()(output_tensor)
    return output_tensor

def dense_add(dim_input, width_list, dim_output,   activation_function=None, regularizer_function = regularizers.l1(0.01),
              kernel_initializer_function = initializers.he_uniform(), bias_initializer_function = initializers.he_normal() ):
    input_tensor = Input(shape=(dim_input,))
    input_list = []
    x0 = Dense(width_list[0], activation=activation_function, kernel_regularizer=regularizer_function,
                       kernel_initializer=kernel_initializer_function, bias_initializer=bias_initializer_function)(input_tensor)
    x0 = BatchNormalization()(x0)
    x0 = PReLU()(x0)
    input_list.append(x0)
    width_len = len(width_list)
    for i in range(1, width_len):
        #if width_list[i] != width_list[i-1]:
            #continue
        if i >1:
            input_temp = Concatenate()(input_list)
        else:
            input_temp = x0
        x_temp = Dense(width_list[i], activation=activation_function, kernel_regularizer=regularizer_function,
                       kernel_initializer=kernel_initializer_function, bias_initializer=bias_initializer_function)(input_temp)
        x_temp = BatchNormalization()(x_temp)
        x_temp = PReLU()(x_temp)
        input_list.append(x_temp)

    output_temp =  Concatenate()(input_list)
    output_tensor = Dense(dim_output, activation=activation_function, kernel_regularizer=regularizer_function,
                       kernel_initializer=kernel_initializer_function, bias_initializer=bias_initializer_function)(output_temp)
    output_tensor = BatchNormalization()(output_tensor)
    output_tensor = PReLU()(output_tensor)
    return input_tensor, output_tensor

def acc(y_true, y_pred):
    y_pred_categ = K.round(y_pred)
    accuracy = K.classification_error(y_true, y_pred_categ)
    return accuracy

from keras.callbacks import LearningRateScheduler

def scheduler(epoch):
    if epoch % 1 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.2)
        print("lr changed to {}".format(lr * 0.2))
    return K.get_value(model.optimizer.lr)



def fcnmodel():
    regularizer_para_1 = regularizers.l1(0.5)
    regularizer_para_2 = regularizers.l2(0.01)
    learning_rate = 0.003
    dim_bhv = 8
    dim_bsc1 = 7
    dim_bsc2 = 739  # 'city prodName color ct carrier'
    dim_act = 9401
    dim_time = 9401
    dim_duration = 9401
    dim_time_new = 9402
    dim_duration_new = 9402
    dim_tfidf =  9401
    dim_app = 56340
    dim_app1 = 40
    dropout_rate = 0.01
    Initializer = initializers.he_uniform()#initializers.TruncatedNormal()
    input_app, encode_app = dense_add(dim_app, [512, 256,  128, 64], 64)
    input_bhv, encode_bhv = dense_add(dim_bhv, [32,  16], 16)
    input_bsc1, encode_bsc1 = dense_add(dim_bsc1, [32,  16], 16)
    input_app1, encode_app2 = dense_add(dim_app1, [32,  16], 32)
    input_bsc2, encode_bsc2 = dense_add(dim_bsc2, [256, 128,  64], 64)
    input_act, encode_act = dense_add(dim_act, [512, 256, 128,  64], 64)
    input_time, encode_time = dense_add(dim_time, [512, 256, 128,  64], 64)
    input_duration, encode_duration = dense_add(dim_duration, [512, 256, 128,  64], 64)
    input_time_new, encode_time_new = dense_add(dim_time_new, [256, 128,  64], 64)
    input_duration_new, encode_duration_new = dense_add(dim_duration_new, [256, 128,  64], 64)
    input_tfidf, encode_tfidf = dense_add(dim_tfidf, [256, 128, 64], 64)
    dim_constant = 886
    input_constant, encode_constant = dense_add(dim_constant, [256, 128, 64], 32)
    dim_cv_max = 5000
    input_cv_max, encode_cv_max = dense_add(dim_cv_max, [256, 128, 64], 64)



    # --- merge ---
    merge = concatenate([encode_app, encode_bhv, encode_bsc1,  encode_bsc2, encode_act, encode_app2,  encode_time, encode_duration,
                         encode_time_new, encode_duration_new, encode_tfidf, encode_constant, encode_cv_max])
    #merge = concatenate([encode_bhv, encode_bsc1, encode_bsc2, encode_act, encode_app2,  encode_time, encode_duration])
    #merge = Dense(256, activation = 'relu')(merge)
    merge = Dense(256, activation=None, kernel_regularizer= regularizer_para_2, kernel_initializer=Initializer,bias_initializer=initializers.he_normal())(merge)
    merge = BatchNormalization()(merge)
    merge = PReLU()(merge)
    #merge_1 = Dropout(dropout_rate)(merge_1)
    #merge = Dense(128, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(merge)
    #merge = BatchNormalizaltion()(merge)
    #merge = Dropout(dropout_rate)(merge)
    merge = Dense(64, activation=None, kernel_regularizer=regularizer_para_2, kernel_initializer=Initializer,
                  bias_initializer=initializers.he_normal())(merge)
    merge = BatchNormalization()(merge)
    merge = PReLU()(merge)

    # output
    out_regress = Dense(1, activation='relu',  name='out_regress')(merge)
    out_clf = Dense(6, activation='softmax', name='out_clf')(merge)

    model = Model(inputs=[input_app, input_bhv, input_bsc1, input_bsc2, input_act, input_app1,  input_time, input_duration,
                          input_time_new, input_duration_new, input_tfidf, input_constant, input_cv_max],
                  outputs=[out_regress, out_clf])
    #model = Model(inputs=[input_bhv, input_bsc1, input_bsc2, input_act, input_app1, input_time, input_duration],outputs=[out_regress, out_clf])

    model.summary()
    #optimizer = optimizers.Adam(lr=learning_rate) # lr=learning_rate
    optimizer = optimizers.RMSprop(lr=learning_rate, decay=1e-5)  # lr=learning_rate

    model.compile(optimizer=optimizer,
                  loss={'out_regress': 'mean_squared_error', 'out_clf': 'categorical_crossentropy'},
                  loss_weights=[1, 100000000000000.],
                  metrics={'out_regress': 'mae', 'out_clf': 'acc'})
    return model


if __name__ == '__main__':


    DATADIR_pro = '../data/'
    #WORKDIR = r'D:\Users\xyliu\digix2019'
    #os.chdir(WORKDIR)

    X_app_6 = sparse.load_npz(DATADIR_pro + '/train_sparse_matrix.npz')


    X_bhv = np.load(DATADIR_pro + '/behav_train.npy')
    X_bhv_log = np.load(DATADIR_pro + '/log_behav_train.npy')

    X_bsc1 = np.load(DATADIR_pro + '/basic1_train.npy')
    X_bsc2 = np.load(DATADIR_pro + '/basic2_train.npy').item()
    X_act = np.load(DATADIR_pro + '/act1hot_train.npy').item()

    #X_app_1 = np.loadtxt(open(DATADIR_pro+"/data_train_appclass.csv", "rb"), delimiter=",", skiprows = 1)
    X_app_1 = pd.read_csv(DATADIR_pro + "/data_train_appclass.csv", sep =",")
    X_app_2 = X_app_1.iloc[:, -40:].values
    X_app = np.array(X_app_2).astype(int)

    X_usage_time = sparse.load_npz(DATADIR_pro + 'data_norm/usage_tm_act_norm_train.npz')

    X_usage_duration = sparse.load_npz(DATADIR_pro + 'data_norm/usage_dur_act_norm_train.npz')

    X_constant = sparse.load_npz(DATADIR_pro + 'present_data/df_train_norm_sparse_876_10_constant.npz')
    X_cv_max = sparse.load_npz(DATADIR_pro + 'present_data/train_actived_cv_max_features_5000.npz')

    X_usage_time_new = sparse.load_npz(DATADIR_pro + 'usage/usageSum_tm_train.npz')

    X_usage_duration_new = sparse.load_npz(DATADIR_pro + 'usage/usageSum_dur_train.npz')
    X_tfidf = sparse.load_npz(DATADIR_pro + 'tfidf/active_tf_idf_train.npz')




    print('X_bhv_log', type(X_bhv_log), X_bhv_log.shape)
    print('X_bsc1', type(X_bsc1), X_bsc1.shape)
    print('X_bsc2', type(X_bsc2), X_bsc2.shape)
    print('X_act', type(X_act), X_act.shape)
    print('X_app', type(X_app), X_app.shape)
    #print('X_app_6', type(X_app_6), X_app_6.shape)
    #print(X_app[0, :])


    Y = np.load(DATADIR_pro + '/age.npy')
    Y_1hot = to_categorical(Y - 1, 6)

    batch_sz = 256
    valid_split = 0.05
    epc = 10


    fname_model = 'model/dnn'
    checkpoint = ModelCheckpoint(fname_model, monitor='val_out_clf_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_out_clf_acc', min_delta=0, patience=2, verbose=1, mode='max')
    #reduce_lr = ReduceLROnPlateau(monitor='val_out_clf_loss', factor=0.1, patience=0, mode='auto')
    reduce_lr = LearningRateScheduler(scheduler)
    model = fcnmodel()

    model.fit(x=[X_app_6, X_bhv_log, X_bsc1, X_bsc2, X_act, X_app,  X_usage_time, X_usage_duration,
                 X_usage_time_new, X_usage_duration_new, X_tfidf, X_constant, X_cv_max ], y=[Y, Y_1hot],
              batch_size=batch_sz, epochs=epc, validation_split=valid_split,
              callbacks=[checkpoint, early_stopping, reduce_lr], verbose=1)  # , class_weight = )
    #model.fit(x=[X_bhv_log, X_bsc1, X_bsc2, X_act, X_app,  X_usage_time, X_usage_duration], y=[Y, Y_1hot],
     #         batch_size=batch_sz, epochs=epc, validation_split=valid_split,
     #         callbacks=[checkpoint, early_stopping], verbose=1)  # , class_weight = )
    #model.save(fname_model)
