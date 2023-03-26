from __future__ import print_function

import argparse
import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential
from pathlib import Path
from datetime import datetime
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test



# data augmentation in tensorflow 1
def datagen():
    datagenerator = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    return datagenerator


def build_vgg(x_shape=(32, 32, 3), num_classes=100):
    """
    VGG16 model architecture
    """

    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def build_vgg_overfit(x_shape=(32, 32, 3), num_classes=100):
    """
    VGG16 model architecture, with no dropout layers
    """
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=x_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    # model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model


def VGG(mode, model_path=None, x_shape=(32, 32, 3), num_classes=10):
    """
    helper function to build/load vgg models

    """
    if mode == 'build':
        print("untrained normal model created")
        model = build_vgg(x_shape, num_classes)
    elif mode == 'load':
        print("loading model from ", model_path)
        model = build_vgg(x_shape, num_classes)
        model.built = True
        model.load_weights(model_path)
    elif mode == 'build_overfit':
        print('untrained overfitting model created')
        model = build_vgg_overfit(x_shape, num_classes)
    elif mode == 'load_overfit':
        print("loading model from ", model_path)
        model = build_vgg_overfit(x_shape, num_classes)
        model.built = True
        model.load_weights(model_path)
    else:
        raise ValueError("Unsupported mode type")
    return model


def logpath(parser):
    """
    Args:
        parser: argument parser passed to the python script
    Returns:
        final_model: the path of the final trained model
        ckpt: the path of the checkpoints of each epoch during training
        tblog: the path of tensorboard logs
    """
    save_path = Path(parser.log_path).joinpath('dp') if parser.dpsgd else Path(parser.log_path).joinpath('nodp')
    save_path = save_path / f"{parser.model}-{parser.dataset}"

    if parser.dpsgd:
        save_path = save_path / \
            f"lr{parser.learning_rate}-noisem{parser.noise_multiplier}-C{parser.l2_norm_clip}-{parser.clip_mode}-{parser.noise_decay_mode}-splitratio{parser.split_ratio}"
    else:
        save_path = save_path / f"lr{parser.learning_rate}-epochs{parser.epochs}-splitratio{parser.split_ratio}"
    save_path = save_path / parser.time_stamp

    final_model = save_path / f"epochs{parser.epochs}"
    ckpt = save_path / 'ckpt'
    tblog = save_path / 'tblog'
    if not final_model.exists():
        final_model.mkdir(parents=True)
    if not ckpt.exists():
        ckpt.mkdir(parents=True)
    if not tblog.exists():
        # os.system(f'rm -r {str(tblog)}')
        tblog.mkdir(parents=True)
    return final_model, ckpt, tblog  # evalmia/logs/dp/vgg-cifar100/lr5e-4-epochs100-noisem1-cclip-nodecay/


def my_callbacks(parser, ckpt=False, tensorboard=False, lr=False):
    """
    callbacks during training.
    """

    lr_drop = 20

    def scheduler(epoch):
        return parser.learning_rate * (0.5 ** (epoch // lr_drop))

    lr_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    _, checkpoint_filepath, tb_logdir = logpath(parser)
    checkpoint_filepath = checkpoint_filepath / "{epoch:03d}"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(tb_logdir), histogram_freq=1)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_filepath),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=False,
        verbose=1)
    callback_ls = []
    if ckpt:
        callback_ls.append(checkpoint_callback)
    if tensorboard:
        callback_ls.append(tensorboard_callback)
    if lr:
        callback_ls.append(lr_scheduler)
    return callback_ls if ckpt or tensorboard or lr else None


def Lenet(num_classes=10, input_size=(32, 32, 3)):
    """
    LeNet architecture
    """
    model = keras.Sequential([
        Conv2D(6, 5,
               strides=1,
               padding='same',
               activation='relu',
               input_shape=input_size),
        AveragePooling2D(2, 2),
        Conv2D(16, 5,
               strides=1,
               padding='valid',
               activation='relu'),
        AveragePooling2D(2, 2),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model


def TfTu(num_classes=10, input_size=(32, 32, 3)):
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_size))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def mnist_model(num_classes=10, input_size=(32, 32, 3)):
    """
    Similar to the LeNet architecture.
    """
    mnist_model = keras.Sequential()
    mnist_model.add(Conv2D(20, (5, 5), input_shape=input_size, activation='relu'))
    mnist_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    mnist_model.add(Conv2D(50, (5, 5), activation='relu'))
    mnist_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    mnist_model.add(Flatten())
    mnist_model.add(Dense(128, activation='relu'))
    mnist_model.add(Dense(num_classes, activation='softmax'))
    return mnist_model


def load_mnist():
    """ load all the mnist data"""
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.mnist.load_data()
    all_data = np.concatenate([train_data, test_data], 0)
    all_labels = np.concatenate([train_labels, test_labels], 0)
    all_data = np.expand_dims(all_data, axis=3)
    all_labels = keras.utils.to_categorical(all_labels, 10)
    return all_data, all_labels


def load_cifar100():
    """ load all the cifar100 data"""
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar100.load_data()
    all_data = np.concatenate([train_data, test_data], 0)
    all_labels = np.concatenate([train_labels, test_labels], 0)
    all_labels = keras.utils.to_categorical(all_labels, 100)
    return all_data, all_labels


def load_cifar10():
    """ load all the cifar10 data"""
    (train_data, train_labels), (test_data, test_labels) = keras.datasets.cifar10.load_data()
    all_data = np.concatenate([train_data, test_data], 0)
    all_labels = np.concatenate([train_labels, test_labels], 0)
    all_labels = keras.utils.to_categorical(all_labels, 10)
    return all_data, all_labels


def load_data(parser):
    """
    Load train/test data by randomly split the original data with a split ratio.
    """
    if parser.dataset == 'mnist':
        all_data, all_labels = load_mnist()
    elif parser.dataset == 'cifar100':
        all_data, all_labels = load_cifar100()
    elif parser.dataset == 'cifar10':
        all_data, all_labels = load_cifar10()
    else:
        raise ValueError('unsupported dataset')
    all_data = all_data.astype('float32')

    if parser.dataset_split_path:   # read from existing indices
        print(f'read indices from {parser.dataset_split_path}')
        with open(parser.dataset_split_path, 'r') as f:
            split_idx = json.load(f)
        train_idx = split_idx['train']
        test_idx = split_idx['test']
    else:
        indices = list(range(len(all_data)))
        np.random.shuffle(indices)

        split = int(np.ceil(parser.split_ratio * len(all_data)))
        train_idx = indices[:split]
        test_idx = indices[split:]
        split_idx = {}
        split_idx['train'] = train_idx
        split_idx['test'] = test_idx
        print(f"writing train/test split indices to {os.path.join(parser.log_path, parser.dataset+'_'+time_stamp()+'.txt')}")
        with open(os.path.join(parser.log_path, parser.dataset+'_'+time_stamp()+'.txt'), 'w+') as f:
            json.dump(split_idx, f)

    train_data = all_data[train_idx]
    test_data = all_data[test_idx]
    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]
    train_data, test_data = normalize(train_data, test_data)
    print('Shape of trainset ', train_data.shape)
    print('Shape of trainlabel ', train_labels.shape)
    print('Shape of testset ', test_data.shape)
    return train_data, train_labels, test_data, test_labels


def load_model(parser, mode='build', model_path=None):
    """
    Load the neural network
    Args:
        parser: argument parser passed to the python script
        mode: if 'build', create an empty model, if 'load', load pretrained weights from model_path
        model_path: the path to load the model
    """
    if parser.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
    if parser.dataset == 'mnist':
        input_size = (28, 28, 1)
    else:
        input_size = (32, 32, 3)
    if mode == 'build':
        if parser.model == 'lenet':
            model = Lenet(num_classes, input_size)
        elif parser.model == 'vgg':
            model = VGG('build', num_classes=num_classes, x_shape=input_size)
        elif parser.model == 'vgg_overfit':
            model = VGG('build_overfit', num_classes=num_classes, x_shape=input_size)
        elif parser.model == 'tftu':
            model = TfTu(num_classes, input_size)
        elif parser.model == 'mnist_model':
            model = mnist_model(num_classes, input_size)
        else:
            raise ValueError('unsupported model type')
    elif mode == 'load' and model_path is not None:
        if parser.model == 'lenet':
            model = Lenet(num_classes, input_size)
            model.built = True
            model.load_weights(model_path)
        elif parser.model == 'vgg':
            model = VGG('load', model_path, num_classes=num_classes, x_shape=input_size)
        elif parser.model == 'vgg_overfit':
            model = VGG('load_overfit', model_path, num_classes=num_classes, x_shape=input_size)
        elif parser.model == 'tftu':
            model = TfTu(num_classes, input_size)
            model.built = True
            model.load_weights(model_path)
        elif parser.model == 'mnist_model':
            model = mnist_model(num_classes, input_size)
            model.built = True
            model.load_weights(model_path)
        else:
            raise ValueError('unsupported model type')
    else:
        raise ValueError('unsupported mode')
    return model


def time_stamp():
    now = datetime.now()
    return f'{now.month:02d}{now.day:02d}{now.hour:02d}{now.minute:02d}'


def probs_to_logits(model):
    """
    Convert the activation function of the last layer of a loaded model to a linear activation (i.e no activation).
    This is helpful if we want to change the pretrained model prediction from probability to logits.
    Args:
        model: loaded TF model

    Returns:
        the model with coverted output layer activation
    """
    model.layers[-1].activation = tf.keras.activations.linear
    model.save('cache.h5')
    return tf.keras.models.load_model('cache.h5')


if __name__ == '__main__':
    parser = parsing()
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        train()
