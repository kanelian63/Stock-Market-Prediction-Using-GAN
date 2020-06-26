import tensorflow as tf
from tensorflow.keras import layers

class generator(tf.keras.models.Model) :
    def __init__(self) :
        super(generator, self).__init__()
        self.lstm = layers.LSTM(128)
        self.sigmoid = layers.Dense(1, activation='sigmoid')
    def call(self, input_data) :


        x = input_data
        x = self.lstm(x)
        x = self.sigmoid(x)

        return x

class discriminator(tf.keras.models.Model) :
    def __init__(self) :
        super(discriminator, self).__init__()
        self.Conv1 = layers.Conv2D(32, (4,1), padding="same", strides=2) 
        self.LReLU1 = layers.LeakyReLU(alpha=0.01) 
        self.BN1 = layers.BatchNormalization()
        self.Conv2 = layers.Conv2D(64, (4,1), padding="same", strides=2) 
        self.LReLU2 = layers.LeakyReLU(alpha=0.01)
        self.BN2 = layers.BatchNormalization()
        self.Conv3 = layers.Conv2D(128, (4,1), padding="same", strides=2) 
        self.LReLU3 = layers.LeakyReLU(alpha=0.01)
        self.BN3 = layers.BatchNormalization()

        self.Flatten = layers.Flatten()
        self.Dense4 = layers.Dense(128)
        self.LReLU4 = layers.LeakyReLU(alpha=0.01)
        self.BN3 = layers.BatchNormalization()
        self.Dense5 = layers.Dense(1, activation="sigmoid")

    def call(self, input_data) :
        x = input_data
        x = self.Conv1(x)
        x = self.LReLU1(x)
        x = self.BN1(x)
        x = self.Conv2(x)
        x = self.LReLU2(x)
        x = self.BN2(x)
        x = self.Conv3(x)
        x = self.LReLU3(x)
        x = self.BN3(x)

        x = self.Flatten(x)
        x = self.Dense4(x)
        x = self.LReLU4(x)
        x = self.Dense5(x)


        return x


class dnn(tf.keras.models.Model) :
    def __init__(self) :
        super(dnn, self).__init__()
        self.Flatten = layers.Flatten()
        self.Dense1 = layers.Dense(256, kernel_initializer='glorot_uniform', activation="relu")
        self.BN1 = layers.BatchNormalization()
        self.Dense2 = layers.Dense(256, kernel_initializer='glorot_uniform', activation="relu")
        self.BN2 = layers.BatchNormalization()
        self.Dense3 = layers.Dense(128, kernel_initializer='glorot_uniform', activation="relu")
        self.BN3 = layers.BatchNormalization()
        self.Dense4 = layers.Dense(128, kernel_initializer='glorot_uniform', activation="relu")
        self.BN4 = layers.BatchNormalization()
        self.Dense5 = layers.Dense(64, kernel_initializer='glorot_uniform', activation="relu")
        self.BN5 = layers.BatchNormalization()
        self.sigmoid = layers.Dense(1, kernel_initializer='glorot_uniform',activation="sigmoid")
    
    def call(self, input_data) :
        x = input_data
        x = self.Flatten(x)
        x = self.Dense1(x)
        x = self.BN1(x)
        x = self.Dense2(x)
        x = self.BN2(x)
        x = self.Dense3(x)
        x = self.BN3(x)
        x = self.Dense4(x)
        x = self.BN4(x)
        x = self.Dense5(x)
        x = self.BN5(x)
        x = self.sigmoid(x)

        return x


class cond_dnn(tf.keras.models.Model) :
    def __init__(self) :
        super(cond_dnn, self).__init__()
        self.Flatten_1 = layers.Flatten()
        self.Flatten_2 = layers.Flatten()


        self.Dense1 = layers.Dense(128, kernel_initializer='glorot_uniform', activation="relu")
        self.BN1 = layers.BatchNormalization()

        self.Dense2 = layers.Dense(128, kernel_initializer='glorot_uniform', activation="relu")
        self.BN2 = layers.BatchNormalization()
        
        self.Dense3 = layers.Dense(128, kernel_initializer='glorot_uniform', activation="relu")
        self.BN3 = layers.BatchNormalization()

        self.Dense4 = layers.Dense(128, kernel_initializer='glorot_uniform', activation="relu")
        self.BN4 = layers.BatchNormalization()


        self.Dense5 = layers.Dense(128, kernel_initializer='glorot_uniform', activation="relu")
        self.BN5 = layers.BatchNormalization()


        self.sigmoid = layers.Dense(1, kernel_initializer='glorot_uniform',activation="sigmoid")
    
    def call(self, input_data) :

        # Feature의 갯수가 22개면, OHLC V, Change, 기타 지표 순
        ohlc = input_data[:,:,:4]
        volume = input_data[:,:,5]
        change = input_data[:,:,6]
        indicator = input_data[:,:,6:]


        x_o = self.Flatten_1(ohlc)
        x_o = self.Dense1(x_o)
        x_o = self.BN1(x_o)

        x_v = self.Dense2(volume)
        x_v = self.BN2(x_v)

        x_c = self.Dense3(change)
        x_c = self.BN3(x_c)

        x_i = self.Flatten_2(indicator)
        x_i = self.Dense4(x_i)
        x_i = self.BN4(x_i)

        x = layers.concatenate([x_o, x_v, x_c, x_i])
        x = self.Dense5(x)
        x = self.BN5(x)


        x = self.sigmoid(x)



        return x


class cnn(tf.keras.models.Model) :
    def __init__(self) :
        super(cnn, self).__init__()
        self.Conv1 = layers.Conv1D(32, 3, kernel_initializer='glorot_uniform', activation='relu', padding="same")
        self.BN1 = layers.BatchNormalization()

        self.Conv2 = layers.Conv1D(32, 3, kernel_initializer='glorot_uniform', activation='relu', padding="same")
        self.BN2 = layers.BatchNormalization()

        self.Conv3 = layers.Conv1D(32, 3, kernel_initializer='glorot_uniform', activation='relu', padding="same")
        self.BN3 = layers.BatchNormalization()

        self.Flatten = layers.Flatten()
        self.sigmoid = layers.Dense(1, activation="sigmoid")

    def call(self, input_data) :

        x = input_data
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        x = self.Conv3(x)
        x = self.BN3(x)
        x = self.Flatten(x)
        x = self.sigmoid(x)

        return x


class rnn(tf.keras.models.Model) :
    def __init__(self) :
        super(rnn, self).__init__()
        self.LSTM = layers.LSTM(32)
        self.sigmoid = layers.Dense(1, activation="sigmoid")

    def call(self, input_data) :

        x = input_data
        x = self.LSTM(x)
        x = self.sigmoid(x)

        return x