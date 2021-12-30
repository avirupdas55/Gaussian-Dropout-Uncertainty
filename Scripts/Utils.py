import h5py
import pickle
import enum
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator

class MyThresholdCallback(Callback):
    def __init__(self, acc_threshold, val_threshold):
        super(MyThresholdCallback, self).__init__()
        self.acc_threshold = acc_threshold
        self.val_threshold= val_threshold

    def on_epoch_end(self, epoch, logs=None): 
        acc = logs["accuracy"]
        val_acc = logs["val_accuracy"]
        if acc>=self.acc_threshold and val_acc >= self.val_threshold:
            self.model.stop_training = True

class DatasetType(enum.Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3

class MonkeyData:
    def __init__(self, path: str, target_size: Tuple[int, int], dataset_type: DatasetType, batch_size: int,
                 data_generator: ImageDataGenerator = None, class_filter: List[str] = None,
                 label_mapping: Dict[int, str] = None):
        self.path = path
        if dataset_type == DatasetType.TRAIN:
            self.dataset_subset = 'training'
            self.dataset_shuffle = True
        elif dataset_type == DatasetType.VAL:
            self.dataset_subset = 'validation'
            self.dataset_shuffle = False
        elif dataset_type == DatasetType.TEST:
            self.dataset_subset = None
            self.dataset_shuffle = False
        self.dataset_type = dataset_type
        self.label_mapping = label_mapping
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_filter = class_filter
        if data_generator is not None:
            self.data_generator = data_generator
        else:
            self.data_generator = self._get_data_generator()
        self.data_iterator = self._get_data_iterator()

    def _get_data_generator(self) -> ImageDataGenerator:
        if self.dataset_type == DatasetType.TRAIN:
            return ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
                                      height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
                                      horizontal_flip=True, fill_mode='nearest', validation_split=0.2)
        else:
            return ImageDataGenerator(rescale=1./255)

    def _get_data_iterator(self) -> DirectoryIterator:
        return self.data_generator.flow_from_directory(self.path, target_size=self.target_size,
                                                       class_mode='categorical',
                                                       subset=self.dataset_subset,
                                                       shuffle=self.dataset_shuffle,
                                                       batch_size=self.batch_size,
                                                       seed=42,
                                                       classes=self.class_filter)

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect all data and label by exhausting the data iterator
        and returns a single array of data and a single array of labels
        :return: ndarray of images, ndarray of labels
        """
        test_num = self.data_iterator.samples

        data, label = [], []
        for i in range((test_num // self.batch_size) + 1):
            x, y = self.data_iterator.next()
            data.append(x)
            label.append(y)

        data = np.vstack(data)
        label = np.argmax(np.vstack(label), axis=1)
        return data, label

def pre_process_CO2(X_train, y_train=None, normalize=True):
  if normalize:
        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        mean_X_train = np.mean(X_train, 0)
  else:
        std_X_train = np.ones(X_train.shape[1])
        mean_X_train = np.zeros(X_train.shape[1])

  X_train = (X_train - np.full(X_train.shape, mean_X_train)) / np.full(X_train.shape, std_X_train)

  if y_train is not None:
    mean_y_train = np.mean(y_train)
    std_y_train = np.std(y_train)
    y_train_normalized = (y_train - mean_y_train) / std_y_train
    
    return X_train,y_train_normalized
  return X_train

def inv_transform(y, std_y, mean_y):
  return y*std_y+mean_y

def get_mnist_data():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    X_train = X_train / 255.
    X_test = X_test / 255.
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (X_train, y_train), (X_test, y_test)

def bayesian_pred(model, test_images, T=2000):
  pred_bayes_dist=[]
  for _ in tqdm(range(0,T)):
    pred_bayes_dist.append(model.predict(test_images))
  #pred_bayes_dist= np.transpose(np.vstack(pred_bayes_dist),(1,0,2))
  return pred_bayes_dist