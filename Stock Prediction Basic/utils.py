import os
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from hyperparameters import hp
import matplotlib.pyplot as plt

def multi_stock_generator(path_dir, batch_size, M, N, 
                            seq_len, n_features, num_stock_size,
                            norm=True) :
    file_list = os.listdir(path_dir)
    file_length = len(file_list)

    while True :
        idx = np.random.randint(file_length)
        stock_batch = np.load(path_dir + "/" + file_list[idx])[20:, :] #for data claerity
        if norm :
            stock_batch = c_minmaxnorm(stock_batch)

        for j in range(num_stock_size) :
            data = stock_batch[N*j: (N*j + M+N)]
                
            yield data

def single_stock_generator(data, batch_size, M, N,
                            seq_len, n_features, num_stock_size, 
                            norm=True) :
    _seq_len = data.shape[0]
    data = data[_seq_len - seq_len:]
                
    while True :        
        for j in range(num_stock_size) :
            _data = data[N*j : (N*j + M+N)]
            _data = c_minmaxnorm(_data, M)

            yield _data
            
def make_batch(data, n_features, _test=False) :
    T = hp.T
    batch_size = hp.batch_size

    x_batch = np.zeros(shape=(batch_size, T, n_features))
    y_batch = np.zeros(shape=(batch_size,))
    
    if _test :
        _L = hp.N
    else :
        _L = hp.M

    for k in range(batch_size) :
        idx = np.random.randint(_L-T-1)
        x_batch[k] = data[idx:idx+T, :]
        y_batch[k] = data[idx+T, 3]
    #import pdb;pdb.set_trace()

    if hp._task == "class" :
        tmp = x_batch[:,-1,3] - y_batch
        y_batch = (tmp>0) - 0

    return x_batch, y_batch

def split_data(data, length, seq_len) :
    '''
    Simply split data into train and test set.
    input
        data : pd.DataFrame
    '''
    train_set = data[:length]
    test_set = data[length-seq_len:]
    return train_set, test_set

def minmaxnorm(data) :
    '''
    Using sklearn package, do min-max normalization.
    Feature range set (0.05, 0.95).
    '''
    #import pdb; pdb.set_trace()
    #data = data.reshape(-1,1)

    scaler = MinMaxScaler(feature_range=(0.05,0.95))
    scaler.fit(data)
    data = scaler.transform(data)
    
    return data, scaler

def c_minmaxnorm(data, 
                M=240, scale_max=1.75, scale_min=0.5) :
    '''
    Custom minmax scaling function.
    When M = 240, N = 60.
    max * 1.75 and min * 0.5.
    

    '''
    _data = data[:M]
    _max = np.max(_data, 0) * scale_max
    _min = np.min(_data, 0) * scale_min
    _data = np.vstack((_data, _max, _min))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(_data)

    data = scaler.transform(data)

    return data

def data_norm(data, n_features, method) :

    if method == "minmax" :
        data_set, data_scaler = minmaxnorm(data)
        #data_set = data_set.reshape(-1, n_features)
    elif method == "sigmoid" :
        data_set = sigmoidnorm(data, n_features)
        data_scaler = 1
    else :
        print("Wrong normalization method!")
    return data_set, data_scaler

def plot_loss(losses) :
    plt.figure(figsize=(10,10))

    plt.plot(losses[0], label="train losses")
    plt.plot(losses[1], label="val losses")
    plt.xlabel("Epochs or ...")
    plt.ylabel("Losses")
    plt.legend()
    plt.show()



# Last Metric

def saveHistory(test_data_gen, model, M, num_stock_size, num_test_iters) :            
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    for i in range(num_stock_size) :
        data_ = next(test_data_gen)
        test_data = data_[M:, :]

        for iter in range(num_test_iters) :
            x_batch, y_batch = make_batch(test_data, hp.n_features, _test=True)
            y_ = model(x_batch)
            if i == 0 and iter == 0 :
                y_true = y_batch
                y_pred = y_
            else : 
                y_true = np.concatenate((y_true, y_batch))
                y_pred = np.concatenate((y_pred, y_))

    #import pdb; pdb.set_trace()
    #y_true = np.argmax(y_true,axis=1)
    y_pred = np.argmax(y_pred,axis=1)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    if tp == 0:
        tp = 1
    if tn == 0:
        tn = 1
    if fp == 0:
        fp = 1
    if fn == 0:
        fn = 1
    TPR = float(tp)/(float(tp)+float(fn))
    FPR = float(fp)/(float(fp)+float(tn))
    accuracy = round((float(tp) + float(tn))/(float(tp) +
                                            float(fp) + float(fn) + float(tn)), 3)
    specitivity = round(float(tn)/(float(tn) + float(fp)), 3)
    sensitivity = round(float(tp)/(float(tp) + float(fn)), 3)
    mcc = round((float(tp)*float(tn) - float(fp)*float(fn))/math.sqrt(
        (float(tp)+float(fp))
        * (float(tp)+float(fn))
        * (float(tn)+float(fp))
        * (float(tn)+float(fn))
    ), 3)
    
    f_output = open('hasilnya_test.txt', 'a')
    f_output.write('=======\n')
    f_output.write('{}epochs_{}batch_model_{}\n'.format(
        hp.num_epochs, hp.batch_size, hp.model_type))
    f_output.write('TN: {}\n'.format(tn))
    f_output.write('FN: {}\n'.format(fn))
    f_output.write('TP: {}\n'.format(tp))
    f_output.write('FP: {}\n'.format(fp))
    f_output.write('TPR: {}\n'.format(TPR))
    f_output.write('FPR: {}\n'.format(FPR))
    f_output.write('accuracy: {}\n'.format(accuracy))
    f_output.write('specitivity: {}\n'.format(specitivity))
    f_output.write("sensitivity : {}\n".format(sensitivity))
    f_output.write("mcc : {}\n".format(mcc))
    f_output.write("{}".format(report))
    f_output.write('=======\n')
    f_output.close()

    print("To save history is done.")