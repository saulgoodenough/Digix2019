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

# initializers.glorot_normal(), keras.initializers.glorot_uniform(), keras.initializers.he_normal()

def dense_add(dim_input, width_list, dim_output,   activation_function=None, regularizer_function = regularizers.l1(0.05),
              kernel_initializer_function = initializers.glorot_uniform(), bias_initializer_function = 'zeros' ):
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
    regularizer_para_2 = regularizers.l2(0.05)
    learning_rate = 0.01
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
    input_bhv, encode_bhv = dense_add(dim_bhv, [32,  16], 16)
    input_bsc1, encode_bsc1 = dense_add(dim_bsc1, [32,  16], 16)
    input_app1, encode_app2 = dense_add(dim_app1, [32,  16], 16)
    input_bsc2, encode_bsc2 = dense_add(dim_bsc2, [256, 128,  64], 64)
    input_act, encode_act = dense_add(dim_act, [512, 256, 128,  64], 64)
    input_time, encode_time = dense_add(dim_time, [256, 128,   64], 64)
    input_duration, encode_duration = dense_add(dim_duration, [256, 128,  64], 64)
    input_time_new, encode_time_new = dense_add(dim_time_new, [256,   128, 64], 64)
    input_duration_new, encode_duration_new = dense_add(dim_duration_new, [256,  128,   64], 64)
    input_tfidf, encode_tfidf = dense_add(dim_tfidf, [256, 128, 64], 64)
    dim_constant = 886
    input_constant, encode_constant = dense_add(dim_constant, [256, 128,  64], 32)

    dim_cv_max = 5000
    input_cv_max, encode_cv_max = dense_add(dim_cv_max, [256,  128,   64], 64)


    # --- merge ---
    merge = concatenate([encode_constant, encode_time_new,encode_duration_new, encode_tfidf,encode_time, encode_duration, encode_act, encode_bhv, encode_bsc1,  encode_bsc2, encode_app2,  encode_cv_max])
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
    merge = Dense(64, activation=None, kernel_regularizer=regularizer_para_2, kernel_initializer=Initializer,
                  bias_initializer=initializers.glorot_normal())(merge)
    merge = BatchNormalization()(merge)
    merge = PReLU()(merge)

    # output
    out_regress = Dense(1, activation='relu',  name='out_regress')(merge)
    out_clf = Dense(6, activation='softmax', name='out_clf')(merge)

    model = Model(inputs=[input_constant, input_time_new, input_duration_new, input_tfidf, input_time, input_duration, input_act, input_bhv, input_bsc1, input_bsc2,  input_app1,    input_cv_max],
                  outputs=[out_regress, out_clf])
    #model = Model(inputs=[input_bhv, input_bsc1, input_bsc2, input_act, input_app1, input_time, input_duration],outputs=[out_regress, out_clf])

    model.summary()
    #optimizer = optimizers.Adam(lr=learning_rate) # lr=learning_rate
    optimizer = optimizers.RMSprop(lr=learning_rate, decay= 2e-4) #lr=learning_rate

    model.compile(optimizer=optimizer,
                  loss={'out_regress': 'mean_squared_error', 'out_clf': 'categorical_crossentropy'},
                  loss_weights=[1, 1000000000000.],
                  metrics={'out_regress': 'mae', 'out_clf': 'acc'})
    return model


if __name__ == '__main__':


    DATADIR_pro = '../data/'
    #WORKDIR = r'D:\Users\xyliu\digix2019'
    #os.chdir(WORKDIR)

    #X_app_6 = sparse.load_npz(DATADIR_pro + '/train_sparse_matrix.npz')
    X_tfidf = sparse.load_npz(DATADIR_pro + 'tfidf/active_tf_idf_train.npz')
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
    X_constant = sparse.load_npz(DATADIR_pro + 'present_data/df_train_norm_sparse_876_10_constant.npz')
    X_cv_max = sparse.load_npz(DATADIR_pro + 'present_data/train_actived_cv_max_features_5000.npz')

    X_usage_time_new = sparse.load_npz(DATADIR_pro + 'present_data/train_sparse_matrix_times.npz').tocsr()
    X_usage_duration_new = sparse.load_npz(DATADIR_pro + 'present_data/train_sparse_matrix_doration.npz').tocsr()


    #print('X_app_6', type(X_app_6), X_app_6.shape)
    #print(X_app[0, :])
    # test data
    X_tfidf_test = sparse.load_npz(DATADIR_pro + 'tfidf/active_tf_idf_test.npz')

    X_bhv_test = np.load(DATADIR_pro + '/behav_test.npy')
    X_bhv_log_test = np.load(DATADIR_pro + '/log_behav_test.npy')

    X_bsc1_temp_test = np.load(DATADIR_pro + '/basic1_test.npy')
    X_app_number_test = np.load(DATADIR_pro + 'present_data/app_actived_number_test.npy')
    X_bsc1_test = np.concatenate((X_bsc1_temp_test, X_app_number_test), axis=1)

    X_bsc2_test = np.load(DATADIR_pro + '/basic2_test.npy').item()
    X_act_test = np.load(DATADIR_pro + '/act1hot_test.npy').item()

    # X_app_1 = np.loadtxt(open(DATADIR_pro+"/data_train_appclass.csv", "rb"), delimiter=",", skiprows = 1)
    X_app_1_test = pd.read_csv(DATADIR_pro + "/data_test_appclass.csv", sep=",")
    X_app_2_test = X_app_1_test.iloc[:, -40:].values
    X_app_test = np.array(X_app_2_test).astype(int)

    X_usage_time_test = sparse.load_npz(DATADIR_pro + 'data_norm/usage_tm_act_norm_test.npz')

    X_usage_duration_test = sparse.load_npz(DATADIR_pro + 'data_norm/usage_dur_act_norm_test.npz')

    X_constant_test = sparse.load_npz(DATADIR_pro + 'present_data/df_test_norm_sparse_876_10_constant.npz')
    X_cv_max_test = sparse.load_npz(DATADIR_pro + 'present_data/test_actived_cv_max_features_5000.npz')

    X_usage_time_new_test = sparse.load_npz(DATADIR_pro + 'present_data/test_sparse_matrix_times.npz').tocsr()
    X_usage_duration_new_test = sparse.load_npz(DATADIR_pro + 'present_data/test_sparse_matrix_doration.npz').tocsr()

    Y = np.load(DATADIR_pro + '/age.npy')
    Y_1hot = to_categorical(Y - 1, 6)

    batch_sz = 1024
    valid_split = 0.01
    epc = 10

    Prob_array = 0
    for i in range(0, 5):
        fname_model = 'model/dnn' + '_' + str(i)
        checkpoint = ModelCheckpoint(fname_model, monitor='val_out_clf_acc', verbose=1, save_best_only=True, mode='max')
        early_stopping = EarlyStopping(monitor='val_out_clf_acc', min_delta=0, patience=2, verbose=1, mode='max')
        reduce_lr = LearningRateScheduler(scheduler)
        model = fcnmodel()
        model.fit(x=[ X_constant, X_usage_time_new,X_usage_duration_new, X_tfidf,X_usage_time, X_usage_duration, X_act, X_bhv_log, X_bsc1, X_bsc2, X_app,   X_cv_max], y=[Y, Y_1hot],
                  batch_size=batch_sz, epochs=epc, validation_split=valid_split,
                  callbacks=[checkpoint, early_stopping, reduce_lr], verbose=1)  # , class_weight = )

        results = model.predict([X_constant_test, X_usage_time_new_test, X_usage_duration_new_test, X_tfidf_test,X_usage_time_test, X_usage_duration_test, X_act_test, X_bhv_log_test, X_bsc1_test, X_bsc2_test, X_app_test,   X_cv_max_test])

        print(len(results))
        proba_array = results[1]

        class_results = np.argmax(proba_array, axis=1) + 1
        print(class_results)
        print(len(class_results))

        x_temp = 1 / np.sum(proba_array, axis=1)
        x_temp = x_temp.reshape((502500, 1))

        proba_array_temp = np.multiply(proba_array, x_temp)
        Prob_array += proba_array_temp
        print(proba_array_temp)
        print(np.shape(proba_array_temp))
        file_name = 'FCN_prob' + '_'+ str(i) + '.npy'
        np.save(file_name, proba_array_temp)
    Prob_array = Prob_array/5.0
    np.save('FCN_prob_average.npy', Prob_array)
    #model.fit(x=[X_bhv_log, X_bsc1, X_bsc2, X_act, X_app,  X_usage_time, X_usage_duration], y=[Y, Y_1hot],
     #         batch_size=batch_sz, epochs=epc, validation_split=valid_split,
     #         callbacks=[checkpoint, early_stopping], verbose=1)  # , class_weight = )
    #model.save(fname_model)
