# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""
Denoise Auto-encoder

Denosing auto encoders are an important and crucial tools for feature selection and extraction.
"""
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class DAE:
    def __init__(self, encoder_units=(500, 500), feature_units=20, activation='relu',
                 kernel_initializer='glorot_uniform', optimizer=Adam(learning_rate=0.001), noise_rate=0):
        self.encoder_units = encoder_units
        self.feature_units = feature_units
        self.activate = activation
        self.kernel_initializer = kernel_initializer
        self.optimizer = optimizer
        self.noise_rate = noise_rate
        return

    def build_dae2(self, X):
        inputs = Input((X.shape[1],))
        x = Dense(100, activation='relu')(inputs)  # 1500 original
        x = Dense(20, activation='relu', name="feature_layer")(x)  # 1500 original
        x = Dense(100, activation='relu')(x)  # 1500 original
        outputs = Dense(X.shape[1], activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
        return model

    def build_dae(self, X):

        # denoising autoencoder
        inputs = Input((X.shape[1],), name='input_layer')
        n_stacks = len(self.encoder_units) - 1
        # input
        x = inputs
        # internal layers in encoder
        for i in range(n_stacks):
            x = Dense(self.encoder_units[i + 1], activation=self.activate, kernel_initializer=self.kernel_initializer,
                      name='encoder_%d' % i)(x)

        # hidden layer
        x = Dense(self.feature_units, kernel_initializer=self.kernel_initializer,
                  name='feature_layer')(x)
        # hidden layer, features are extracted from here

        # internal layers in decoder
        for i in range(n_stacks, 0, -1):
            x = Dense(self.encoder_units[i], activation=self.activate, kernel_initializer=self.kernel_initializer,
                      name='decoder_%d' % i)(x)

        # output
        x = Dense(X.shape[1], activation=self.activate, kernel_initializer=self.kernel_initializer,
                  name='output_layer')(x)
        output = x
        return Model(inputs=inputs, outputs=output, name='AE')

    def fit(self, X, batch_size=128, epochs=1000):
        es = EarlyStopping(monitor='mse', min_delta=0.001, patience=5,
                           verbose=1, mode='min', baseline=None, restore_best_weights=True)

        rlr = ReduceLROnPlateau(monitor='mse', factor=0.5,
                                patience=3, min_lr=1e-6, mode='min', verbose=1)

        autoencoder = self.build_dae(X)
        # autoencoder.compile(optimizer=self.optimizer, loss='mse',metrics=['mse'])
        autoencoder.compile(optimizer=self.optimizer, loss='mse', metrics=['mse'])

        if self.noise_rate <= 0:
            print('no noise.')
            autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, callbacks=[es, rlr])
        else:
            print(f'noise rate:{self.noise_rate}')

            gen = self.mix_generator(X, batch_size, swaprate=self.noise_rate)
            autoencoder.fit_generator(generator=gen,
                                      steps_per_epoch=np.ceil(X.shape[0] / batch_size),
                                      epochs=epochs,
                                      callbacks=[es, rlr],
                                      verbose=1,
                                      )
        return autoencoder

    def fit_transform(self, X, batch_size=128, epochs=1000):
        ae = self.fit(X, batch_size, epochs)
        proxy_model = self.__buld_proxy_model(ae, 'feature_layer')
        features = proxy_model.predict(X, batch_size=batch_size)
        return features

    def __buld_proxy_model(self, model, output_layer):
        model.trainable = False
        output = model.get_layer(output_layer).output
        proxy = Model(inputs=model.input, outputs=output)
        return proxy

    def x_generator(self, x, batch_size, shuffle=True):
        # batch generator of input
        batch_index = 0
        n = x.shape[0]
        while True:
            if batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)
            current_index = (batch_index * batch_size) % n
            # print("current_index:{}".format(current_index))

            if n >= current_index + batch_size:
                current_batch_size = batch_size
                batch_index += 1
            else:
                current_batch_size = n - current_index
                batch_index = 0
            batch_x = x[index_array[current_index: current_index + current_batch_size]]

            yield batch_x

    def mix_generator(self, x, batch_size, swaprate=0.15, shuffle=True):
        # generator of noized input and output
        # swap 0.15% of values of datasets with values of another
        num_value = x.shape[1]
        # print("X.shape[1]={}, x.shape[1]={}".format(X.shape[1], x.shape[1]))
        num_swap = int(num_value * swaprate)
        gen1 = self.x_generator(x, batch_size, shuffle)
        gen2 = self.x_generator(x, batch_size, shuffle)
        while True:
            batch1 = next(gen1)
            batch2 = next(gen2)
            new_batch = batch1.copy()
            for i in range(batch1.shape[0]):
                swap_idx = np.random.choice(num_value, num_swap, replace=False)
                new_batch[i, swap_idx] = batch2[i, swap_idx]

            yield (new_batch, batch1)
