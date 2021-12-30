import numpy as np

import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, Activation, Flatten
from tensorflow.keras.regularizers import L2
from tensorflow.keras.optimizers import Adam, Nadam, SGD

from CustomDropout import *
from Utils import *

class CO2_MLP(Model):
  def __init__(self, hidden, drop_rate, activation, shape, length_scale, tau, gaussian=False):
    super(CO2_MLP, self).__init__()
    self.reg = length_scale**2 *(1-drop_rate)/(2. *shape[0]*tau)
    #self.reg=1e-6
    self.model= Sequential()
    self.model.add(Input(shape=(shape[1], )))
    self.model.add(Dense(hidden[0], activation= activation, use_bias=True,
                         kernel_regularizer=L2(self.reg), bias_regularizer=L2(self.reg)))
    
    for i in hidden[1:]:
      if gaussian:
        self.model.add(myGaussianDropout(drop_rate, training=True))
      else:
        self.model.add(myDropout(drop_rate, training=True))
      self.model.add(Dense(i, activation= activation, use_bias=True,
                         kernel_regularizer=L2(self.reg), bias_regularizer=L2(self.reg)))
    if gaussian:
      self.model.add(myGaussianDropout(drop_rate, training=True))
    else:
      self.model.add(myDropout(drop_rate, training=True))
    self.model.add(Dense(1, kernel_regularizer=L2(self.reg)))
    
  def call(self, inputs):
    return self.model(inputs)

class CO2_regressor:
  def __init__(self, hidden, drop_rate, activation, shape, gaussian=False):
    super(CO2_regressor, self).__init__()
    self.length_scale= 1e-2
    self.tau= 0.427114830213
    self.model= CO2_MLP(hidden, drop_rate, activation, shape, self.length_scale, self.tau, gaussian=gaussian)

  def train(self, x, y, batch_size=32, lr=0.001, max_epoch=1000000, callbacks=None, verbose=True):
    opt= Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    #opt= SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    history= self.model.fit(x,y, batch_size=batch_size, epochs=max_epoch, callbacks=callbacks, verbose=verbose)

    return history

  def summary(self):
    self.model.summary()

  def get_predictions(self, X, T=10000):
    Yt_hat = np.array([self.model.predict(X) for _ in range(T)]).squeeze()
    return Yt_hat

  def save(self, path):
    self.model.save(path)
    
class MNISTClassifier:
  
  def __init__(self, drop_rate, gaussian=False):

    super(MNISTClassifier, self).__init__()
    self.drop_rate=drop_rate
    LeNet= Sequential()

    # First Block
    LeNet.add(Conv2D(filters=20, kernel_size=(5,5), padding='same', input_shape=(28,28,1)))
    if gaussian:
      LeNet.add(myGaussianDropout(self.drop_rate, training=True))
    else:
      LeNet.add(myDropout(self.drop_rate, training=True))
    LeNet.add(MaxPool2D(pool_size=(2,2), strides=2))

    # Second Block
    LeNet.add(Conv2D(filters=50, kernel_size=(5,5), padding='same'))
    if gaussian:
      LeNet.add(myGaussianDropout(self.drop_rate, training=True))
    else:
      LeNet.add(myDropout(self.drop_rate, training=True))
    LeNet.add(MaxPool2D(pool_size=(2,2), strides=2))
    
    # Final Block
    LeNet.add(Flatten())
    LeNet.add(Dense(units=500, activation='relu'))
    if gaussian:
      LeNet.add(myGaussianDropout(self.drop_rate, training=True))
    else:
      LeNet.add(myDropout(self.drop_rate, training=True))
    LeNet.add(Dense(10))
    LeNet.add(Activation('softmax'))
    self.model= LeNet

  def train(self, X_train, y_train, X_test, y_test, batch_size=64, lr=0.01, max_epoch=300, callbacks=None, verbose=True):
    opt= Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history= self.model.fit(X_train, y_train, batch_size=batch_size, validation_data=(X_test,y_test),
                            epochs=max_epoch, callbacks=callbacks, verbose=verbose)

    return history

  def summary(self):
    self.model.summary()

  def get_predictions(self, X, T=10000):
    standard_pred= np.argmax(self.model.predict(X), axis=-1)
    softmax_input_model = Model(self.model.inputs, self.model.layers[-2].output)
    softmax_output_model = Model(self.model.inputs, self.model.layers[-1].output)

    y1 = np.array([softmax_input_model(X) for _ in range(T)]).squeeze()
    y2 = np.array([softmax_output_model(X) for _ in range(T)]).squeeze()
    return standard_pred, y1, y2

  def save(self, path):
    self.model.save(path) 


class MonkeyClassifier2:
  
  def __init__(self, num_classes, drop_rate, gaussian=False):

    super(MonkeyClassifier2, self).__init__()
    self.drop_rate_1=drop_rate[0]
    self.drop_rate_2=drop_rate[1]
    MonkeyModel=Sequential()

    # First Block
    MonkeyModel.add(Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', input_shape=(150,150,3)))
    MonkeyModel.add(MaxPool2D(pool_size=(2,2)))
    if gaussian:
      MonkeyModel.add(myGaussianDropout(self.drop_rate_1, training=True))
    else:
      MonkeyModel.add(myDropout(self.drop_rate_1, training=True))

    # Second Block
    MonkeyModel.add(Conv2D(filters=32, kernel_size=(3,3), activation = 'relu'))
    MonkeyModel.add(MaxPool2D(pool_size=(2,2)))
    if gaussian:
      MonkeyModel.add(myGaussianDropout(self.drop_rate_1, training=True))
    else:
      MonkeyModel.add(myDropout(self.drop_rate_1, training=True))

    # Third Block
    MonkeyModel.add(Conv2D(filters=64, kernel_size=(3,3), activation = 'relu', padding='same'))
    MonkeyModel.add(Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'))
    MonkeyModel.add(MaxPool2D(pool_size=(2,2)))
    if gaussian:
      MonkeyModel.add(myGaussianDropout(self.drop_rate_1, training=True))
    else:
      MonkeyModel.add(myDropout(self.drop_rate_1, training=True))
    
    # Final Block
    MonkeyModel.add(Flatten())
    MonkeyModel.add(Dense(units=512, activation='relu'))
    if gaussian:
      MonkeyModel.add(myGaussianDropout(self.drop_rate_2, training=True))
    else:
      MonkeyModel.add(myDropout(self.drop_rate_2, training=True))
    MonkeyModel.add(Dense(num_classes))
    MonkeyModel.add(Activation('softmax'))
    
    self.model= MonkeyModel

  def train(self, train_dataset: MonkeyData, val_dataset: MonkeyData, lr=0.01, max_epoch=200, callbacks=None, verbose=True):
    opt= Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)

    train_steps = train_dataset.data_iterator.samples // train_dataset.data_iterator.batch_size
    val_steps = val_dataset.data_iterator.samples // val_dataset.data_iterator.batch_size

    self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history= self.model.fit(train_dataset.data_iterator, steps_per_epoch=train_steps,
                            validation_data=val_dataset.data_iterator, validation_steps=val_steps,
                            epochs=max_epoch, callbacks=callbacks, verbose=verbose)

    return history

  def summary(self):
    self.model.summary()

  def get_predictions(self, test_dataset: MonkeyData) -> np.ndarray:
    return self.model.predict(test_dataset.data_iterator)

  def save(self, path):
    self.model.save(path)     


class MonkeyClassifier:
  
  def __init__(self, num_classes, drop_rate, gaussian=False):

    super(MonkeyClassifier, self).__init__()
    self.drop_rate_1=drop_rate[0]
    self.drop_rate_2=drop_rate[1]
    MonkeyModel=Sequential()

    # First Block
    MonkeyModel.add(Conv2D(filters=32, kernel_size=(3,3), activation = 'relu', input_shape=(224,224,3)))
    MonkeyModel.add(MaxPool2D(pool_size=(2,2)))
    MonkeyModel.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
    MonkeyModel.add(MaxPool2D(pool_size=(2,2)))
    if gaussian:
      MonkeyModel.add(myGaussianDropout(self.drop_rate_1, training=True))
    else:
      MonkeyModel.add(myDropout(self.drop_rate_1, training=True))

    # Second Block
    MonkeyModel.add(Conv2D(filters=64, kernel_size=(3,3), activation = 'relu'))
    MonkeyModel.add(MaxPool2D(pool_size=(2,2)))
    MonkeyModel.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    MonkeyModel.add(MaxPool2D(pool_size=(2,2)))
    if gaussian:
      MonkeyModel.add(myGaussianDropout(self.drop_rate_1, training=True))
    else:
      MonkeyModel.add(myDropout(self.drop_rate_1, training=True))
    
    # Final Block
    MonkeyModel.add(Flatten())
    MonkeyModel.add(Dense(units=512, activation='relu'))
    if gaussian:
      MonkeyModel.add(myGaussianDropout(self.drop_rate_2, training=True))
    else:
      MonkeyModel.add(myDropout(self.drop_rate_2, training=True))
    MonkeyModel.add(Dense(num_classes))
    MonkeyModel.add(Activation('softmax'))
    
    self.model= MonkeyModel

  def train(self, train_dataset: MonkeyData, val_dataset: MonkeyData, lr=0.01, max_epoch=200, callbacks=None, verbose=True):
    opt= Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)

    train_steps = train_dataset.data_iterator.samples / train_dataset.data_iterator.batch_size
    val_steps = val_dataset.data_iterator.samples / val_dataset.data_iterator.batch_size

    self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history= self.model.fit(train_dataset.data_iterator, steps_per_epoch=train_steps,
                            validation_data=val_dataset.data_iterator, validation_steps=val_steps,
                            epochs=max_epoch, callbacks=callbacks, verbose=verbose)

    return history

  def summary(self):
    self.model.summary()

  def get_predictions(self, test_dataset: MonkeyData) -> np.ndarray:
    return self.model.predict(test_dataset.data_iterator)

  def save(self, path):
    self.model.save(path)     