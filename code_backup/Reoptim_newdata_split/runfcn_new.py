#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras.backend as K
from keras import regularizers
from keras.layers import Input, Dense, concatenate, BatchNormalization, Dropout,Embedding, Reshape, LSTM, \
    Add, Multiply, Average,Concatenate, PReLU, Lambda, Flatten
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

def dense_split(dim_input, dim_output_block, block_num, block_width_list = [128, 64, 32, 8], regularizer_function = regularizers.l1(0.01)):
    last_block_dim = dim_input % block_num
    average_block_dim = dim_input // block_num
    input_tensor = Input(shape=(dim_input,))
    input_list = []
    output_list = []
    for i in range(block_num-1):
        #input_tensor_temp = Input(shape=(average_block_dim,))
        input_tensor_temp = Lambda(lambda x: x[i*average_block_dim:i*average_block_dim+average_block_dim])(input_tensor)
        output_tensor_temp = dense_add_update(input_tensor_temp, block_width_list, dim_output_block, regularizer_function = regularizer_function)
        output_list.append(output_tensor_temp)
    if last_block_dim != 0:
        input_tensor_temp = Lambda(lambda x: x[(block_num - 1) * average_block_dim:dim_input])(input_tensor)
        output_tensor_temp = dense_add_update(input_tensor_temp, block_width_list, dim_output_block, regularizer_function = regularizer_function)
        output_list.append(output_tensor_temp)
    output_tensor = Concatenate(axis=0)(output_list)
    #output_tensor = Flatten()(output_tensor)
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
    dropout_rate = 0.01
    Initializer = initializers.he_uniform()#initializers.TruncatedNormal()

    dim_cont = 56
    input_cont, encode_cont = dense_add(dim_cont, [64, 32, 16], 16)
    dim_categ = 737
    input_categ, encode_categ = dense_add(dim_categ, [128,  64, 32], 32)
    dim_active = 5000
    input_active, encode_active = dense_split(dim_input = dim_active, dim_output_block=2, block_num=48, block_width_list = [32, 8], regularizer_function = None)
    encode_active = dense_add_update(encode_active, [256, 128,  64], 64)
    dim_time = 5000
    input_time, encode_time = dense_split(dim_input=dim_time, dim_output_block=2, block_num=48, block_width_list=[32, 8], regularizer_function = None)
    encode_time = dense_add_update(encode_time, [256, 128,  64], 64)
    dim_duration = 5000
    input_duration, encode_duration = dense_split(dim_input=dim_duration, dim_output_block=2, block_num=48, block_width_list=[32, 8], regularizer_function = None)
    encode_duration = dense_add_update(encode_duration, [256, 128,  64], 64)


    # --- merge ---
    merge = concatenate([encode_cont, encode_categ, encode_active, encode_time, encode_duration])
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

    model = Model(inputs=[input_cont, input_categ, input_active, input_time, input_duration],
                  outputs=[out_regress, out_clf])
    #model = Model(inputs=[input_bhv, input_bsc1, input_bsc2, input_act, input_app1, input_time, input_duration],outputs=[out_regress, out_clf])

    model.summary()
    #optimizer = optimizers.Adam(lr=learning_rate) # lr=learning_rate
    optimizer = optimizers.RMSprop(lr=learning_rate, decay=1e-5)  #1 lr=learning_rate

    model.compile(optimizer=optimizer,
                  loss={'out_regress': 'mean_squared_error', 'out_clf': 'categorical_crossentropy'},
                  loss_weights=[1, 100000000000000.],
                  metrics={'out_regress': 'mae', 'out_clf': 'acc'})
    return model


if __name__ == '__main__':


    DATADIR_pro = '../data/data_20190817/'
    #WORKDIR = r'D:\Users\xyliu\digix2019'
    #os.chdir(WORKDIR)
    # 首先读取所有的信息
    data_train = pd.read_csv(DATADIR_pro+"data_train.csv")
    data_test = pd.read_csv(DATADIR_pro+"data_test.csv")
    # 将里面的分类属性去除
    names_categ = 'city prodName color ct carrier'.split()
    names_cont = list(set(data_train.columns.tolist()) - set(names_categ) - set(['uid', 'label']))
    # 读取连续的属性
    train_x_cont = np.array(data_train[names_cont].values)
    test_x_cont = np.array(data_test[names_cont].values)
    # 读取离散的属性
    train_x_categ = sparse.load_npz(DATADIR_pro+"x_train_categ.npz") # 737
    test_x_categ = sparse.load_npz(DATADIR_pro+"x_test_categ.npz")
    # actived里面出现频率前面5000个app
    X_act_cv = sparse.load_npz(DATADIR_pro+"train_actived_cv_max_features_5000.npz")
    X_act_cv_test = sparse.load_npz(DATADIR_pro+"test_actived_cv_max_features_5000.npz")
    # 根据actived里面5000个app，提取每一个app在30天内的总使用时间和总使用次数
    train_sparse_matrix_doration_npz = sparse.load_npz(DATADIR_pro+"train_sparse_matrix_duration_sum.npz")
    test_sparse_matrix_doration_npz = sparse.load_npz(DATADIR_pro+"test_sparse_matrix_duration_sum.npz")
    train_sparse_matrix_times_npz = sparse.load_npz(DATADIR_pro+"train_sparse_matrix_times_sum.npz")
    test_sparse_matrix_times_npz = sparse.load_npz(DATADIR_pro+"test_sparse_matrix_times_sum.npz")
    print(np.shape(train_x_cont))
    print(np.shape(train_x_categ))
    print(np.shape(test_x_categ))
    print(np.shape(train_sparse_matrix_doration_npz))
    print(np.shape(test_sparse_matrix_doration_npz))
    print(np.shape(train_sparse_matrix_times_npz))
    print(np.shape(test_sparse_matrix_times_npz))



    Y = np.load('../data/age.npy')
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

    model.fit(x=[train_x_cont, train_x_categ, X_act_cv, train_sparse_matrix_doration_npz, train_sparse_matrix_times_npz], y=[Y, Y_1hot],
              batch_size=batch_sz, epochs=epc, validation_split=valid_split,
              callbacks=[checkpoint, early_stopping, reduce_lr], verbose=1)  # , class_weight = )
    #model.fit(x=[X_bhv_log, X_bsc1, X_bsc2, X_act, X_app,  X_usage_time, X_usage_duration], y=[Y, Y_1hot],
     #         batch_size=batch_sz, epochs=epc, validation_split=valid_split,
     #         callbacks=[checkpoint, early_stopping], verbose=1)  # , class_weight = )
    #model.save(fname_model)
