import tensorflow as tf
import numpy as np
from tqdm import tqdm


import models
import utils
from hyperparameters import hp
from utils import make_batch

def train_step(train_data_gen,
                    num_epochs, num_stock_size, num_iters) : 

    if hp._gpu :
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for k in range(len(physical_devices)):
                tf.config.experimental.set_memory_growth(physical_devices[k], True)
                print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
        else:
            print("Not enough GPU hardware devices available")



    M = hp.M

    optimizer = tf.keras.optimizers.Adam(learning_rate = hp.lr)
    if hp.model_type == "dnn" :
        model = models.dnn()
    elif hp.model_type == "cnn" :
        model = models.cnn()
    elif hp.model_type == "rnn" :
        model = models.rnn()
    elif hp.model_type == "cond_dnn" :
        model = models.cond_dnn()
    else :
        print("model type error")
    if hp._task == "class" :
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    else :
        loss_fn = tf.keras.losses.MeanSquaredError()

    #model.summary()

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs) :
        epoch_loss_avg = tf.keras.metrics.Mean()
        val_loss_avg = tf.keras.metrics.Mean()

        for i in range(num_stock_size) :
            data_ = next(train_data_gen) #data_[M+N]
            train_data = data_[:M,:]
            test_data = data_[M:, :]

            for j in range(num_iters) :
                x_batch, y_batch = make_batch(train_data, hp.n_features)
                
                with tf.GradientTape() as tape :
                    y_ = model(x_batch)
                    loss_value = loss_fn(y_batch, y_)
                    grads = tape.gradient(loss_value, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss_avg(loss_value)
               
            num_test_iters = num_iters // 4
            for j in range(num_test_iters) :
                x_batch, y_batch = make_batch(test_data, hp.n_features, _test=True)
                y_ = model(x_batch)
                val_loss_value = loss_fn(y_batch, y_)

                val_loss_avg(val_loss_value)

        train_losses.append(loss_value)
        val_losses.append(val_loss_value)
        
        print("Epoch {:03d}: , Number of stock time {:03d}, Loss: {:.5f}".format(epoch, i,
                                                                                epoch_loss_avg.result()))
        print("Val_Loss: {:.3f}".format(val_loss_avg.result()))

    if hp.b_loss_plot :
        losses = [train_losses, val_losses]
        utils.plot_loss(losses)


    return model