#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
import tensorflow as tf

def focal_loss(gamma=2., alpha=4.):

    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})

        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.log(model_out))
        weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)
    return focal_loss_fixed



def binary_focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    def categorical_focal_loss_fixed(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


# initializers.glorot_normal(), keras.initializers.glorot_uniform(), keras.initializers.he_normal()

def dense_add(dim_input, width_list, dim_output,   activation_function=None, regularizer_function = regularizers.l2(0.5),
              kernel_initializer_function = initializers.glorot_uniform(), bias_initializer_function = initializers.glorot_normal() ):
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

def fcnmodel():
    regularizer_para_1 = regularizers.l1(0.5)
    regularizer_para_2 = regularizers.l2(0.4)
    learning_rate = 0.002
    dim_bhv = 8
    dim_bsc1 = 7+1
    dim_bsc2 = 739  # 'city prodName color ct carrier'
    dim_act = 9401
    dim_time = 9401
    dim_duration = 9401
    dim_app = 56340
    dim_app1 = 40
    dim_time_new = 5000
    dim_duration_new = 5000
    dim_tfidf = 9401
    dropout_rate = 0.01
    Initializer = initializers.glorot_uniform()#initializers.TruncatedNormal()
    #input_app, encode_app = dense_add(dim_app, [512, 256, 256, 128, 128, 128], 64)
    input_bhv, encode_bhv = dense_add(dim_bhv, [32, 32, 16, 16], 16)
    input_bsc1, encode_bsc1 = dense_add(dim_bsc1, [32, 32, 16, 16], 16)
    input_app1, encode_app2 = dense_add(dim_app1, [32, 32,  16, 16], 16)
    input_bsc2, encode_bsc2 = dense_add(dim_bsc2, [256,  128, 128, 64, 64,  64], 64)
    input_act, encode_act = dense_add(dim_act, [512, 256,   128, 128, 64,  64,   64], 64)
    input_time, encode_time = dense_add(dim_time, [512, 128, 128, 64,  64], 64)
    input_duration, encode_duration = dense_add(dim_duration, [512, 128, 128,64,   64], 64)
    #input_time_new, encode_time_new = dense_add(dim_time_new, [256,  128, 128, 64,   64], 64)
    #input_duration_new, encode_duration_new = dense_add(dim_duration_new, [256,  128, 128, 64, 64,  64], 64)
    #input_tfidf, encode_tfidf = dense_add(dim_tfidf, [512, 128, 128, 64, 64], 64)
    #dim_constant = 886
    #input_constant, encode_constant = dense_add(dim_constant, [256, 128, 128,  64, 64], 32)

    dim_cv_max = 5000
    input_cv_max, encode_cv_max = dense_add(dim_cv_max, [512, 256, 128,  128, 64,  64,  64], 64)






    # --- merge ---
    merge = concatenate([ encode_act, encode_bhv, encode_bsc1,  encode_bsc2, encode_app2, encode_time, encode_duration, encode_cv_max])
    #merge = concatenate([encode_bhv, encode_bsc1, encode_bsc2, encode_act, encode_app2,  encode_time, encode_duration])
    #merge = Dense(256, activation = 'relu')(merge)
    merge = Dense(256, activation=None, kernel_regularizer= regularizer_para_2, kernel_initializer=Initializer,bias_initializer=initializers.glorot_normal())(merge)
    merge = BatchNormalization()(merge)
    merge = PReLU()(merge)
    #merge_1 = Dropout(dropout_rate)(merge_1)
    #merge = Dense(128, activation='relu', kernel_regularizer=regularizer_para_2, kernel_initializer='random_normal',bias_initializer='zeros')(merge)
    #merge = BatchNormalizaltion()(merge)
    #merge = Dropout(dropout_rate)(merge)
    #merge = Dense(128, activation=None, kernel_regularizer=regularizer_para_2, kernel_initializer=Initializer,
     #             bias_initializer=initializers.glorot_normal())(merge)
    #merge = BatchNormalization()(merge)
    #merge = PReLU()(merge)
    #merge = Dense(128, activation=None, kernel_regularizer=regularizer_para_2, kernel_initializer=Initializer,
    #              bias_initializer=initializers.glorot_normal())(merge)
    #merge = BatchNormalization()(merge)
    #merge = PReLU()(merge)
    merge = Dense(64, activation=None, kernel_regularizer=regularizer_para_2, kernel_initializer=Initializer,
                  bias_initializer=initializers.glorot_normal())(merge)
    merge = BatchNormalization()(merge)
    merge = PReLU()(merge)

    # output
    out_regress = Dense(1, activation='relu',  name='out_regress')(merge)
    out_clf = Dense(6, activation='softmax', name='out_clf')(merge)

    model = Model(inputs=[ input_act, input_bhv, input_bsc1, input_bsc2,  input_app1, input_time, input_duration,   input_cv_max],
                  outputs=[out_regress, out_clf])
    #model = Model(inputs=[input_bhv, input_bsc1, input_bsc2, input_act, input_app1, input_time, input_duration],outputs=[out_regress, out_clf])

    model.summary()
    optimizer = optimizers.Adam(lr=learning_rate) # lr=learning_rate
    #optimizer = optimizers.RMSprop(lr=learning_rate) #lr=learning_rate

    model.compile(optimizer=optimizer,
                  loss={'out_regress': 'mean_squared_error', 'out_clf':categorical_focal_loss(gamma=1., alpha=1.5)},
                  loss_weights=[1, 1000000000000.],
                  metrics={'out_regress': 'mae', 'out_clf': 'acc'})
    return model


if __name__ == '__main__':


    DATADIR_pro = '../data/'
    #WORKDIR = r'D:\Users\xyliu\digix2019'
    #os.chdir(WORKDIR)

    #X_app_6 = sparse.load_npz(DATADIR_pro + '/train_sparse_matrix.npz')
    #X_tfidf = sparse.load_npz(DATADIR_pro + 'tfidf/active_tf_idf_train.npz')


    X_bhv = np.load(DATADIR_pro + '/behav_train.npy')
    X_bhv_log = np.load(DATADIR_pro + '/log_behav_train.npy')

    X_bsc1_temp = np.load(DATADIR_pro + '/basic1_train.npy')
    X_app_number = np.load(DATADIR_pro + 'present_data/app_actived_number_train.npy')
    X_bsc1 = np.concatenate((X_bsc1_temp, X_app_number), axis=1)

    X_bsc2 = np.load(DATADIR_pro + '/basic2_train.npy').item()
    X_act = np.load(DATADIR_pro + '/act1hot_train.npy').item()

    #X_app_1 = np.loadtxt(open(DATADIR_pro+"/data_train_appclass.csv", "rb"), delimiter=",", skiprows = 1)
    X_app_1 = pd.read_csv(DATADIR_pro + "/data_train_appclass.csv", sep =",")
    X_app_2 = X_app_1.iloc[:, -40:].values
    X_app = np.array(X_app_2).astype(int)

    X_usage_time = sparse.load_npz(DATADIR_pro + 'data_norm/usage_tm_act_norm_train.npz')

    X_usage_duration = sparse.load_npz(DATADIR_pro + 'data_norm/usage_dur_act_norm_train.npz')

    #X_constant = sparse.load_npz(DATADIR_pro + 'present_data/df_train_norm_sparse_876_10_constant.npz')
    X_cv_max = sparse.load_npz(DATADIR_pro + 'present_data/train_actived_cv_max_features_5000.npz')

    #X_usage_time_new = sparse.load_npz(DATADIR_pro + 'present_data/train_sparse_matrix_times.npz').tocsr()
    #X_usage_duration_new = sparse.load_npz(DATADIR_pro + 'present_data/train_sparse_matrix_doration.npz').tocsr()

    print('X_bhv_log', type(X_bhv_log), X_bhv_log.shape)
    print('X_bsc1', type(X_bsc1), X_bsc1.shape)
    print('X_bsc2', type(X_bsc2), X_bsc2.shape)
    print('X_act', type(X_act), X_act.shape)
    print('X_app', type(X_app), X_app.shape)
    #print('X_app_6', type(X_app_6), X_app_6.shape)
    #print(X_app[0, :])


    Y = np.load(DATADIR_pro + '/age.npy')
    Y_1hot = to_categorical(Y - 1, 6)

    batch_sz = 512
    valid_split = 0.05
    epc = 10

    fname_model = 'model/dnn'
    checkpoint = ModelCheckpoint(fname_model, monitor='val_out_clf_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_out_clf_acc', min_delta=0, patience=2, verbose=1, mode='max')
    model = fcnmodel()
    model.fit(x=[ X_act, X_bhv_log, X_bsc1, X_bsc2, X_app, X_usage_time, X_usage_duration,  X_cv_max], y=[Y, Y_1hot],
              batch_size=batch_sz, epochs=epc, validation_split=valid_split,
              callbacks=[checkpoint, early_stopping], verbose=1)  # , class_weight = )
    #model.fit(x=[X_bhv_log, X_bsc1, X_bsc2, X_act, X_app,  X_usage_time, X_usage_duration], y=[Y, Y_1hot],
     #         batch_size=batch_sz, epochs=epc, validation_split=valid_split,
     #         callbacks=[checkpoint, early_stopping], verbose=1)  # , class_weight = )
    #model.save(fname_model)
