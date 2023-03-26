"""
This script runs the Shadow model MIA against a pretrained model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append('./mia/')
import os
from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar100
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
from utils import *
from pathlib import Path
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_paths', nargs='+', help='the paths of vgg models to be attacked, can be multiple paths')
    parser.add_argument('--model', type=str, default='mnist_model',help='vgg/lenet/mnist_model/tftu')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist/cifar100')
    parser.add_argument('--num_shadows', type=int, default="4", help='number of shadows models to train')
    parser.add_argument('--shadow_dataset_size', type=int, default="10000", help='size of the trainset of shadow model')
    parser.add_argument('--shadow_epoch', type=int, default="100", help='number of epochs the shadow models trained on')
    parser.add_argument('--shadow_lr', type=float, default="5e-4", help='learning rate while training the shadow model')
    parser.add_argument('--attack_epoch', type=int, default="50", help='number of epochs the attacker trained on')
    parser.add_argument('--attack_test_dataset_size', type=int, default="10000", help='size of data_in /data_out to test the attacker')
    parser.add_argument('--log_path', type=str, default='logs', help='the directory to store all the results')
    parser.add_argument('--dataset_split_path', type=str, default=None,
                        help='path to the file which stores the previously randomly split train/test data')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    args = parser.parse_args()
    return args




def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.
    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = keras.models.Sequential()

    model.add(keras.layers.Dense(128, activation="relu", input_shape=(NUM_CLASSES,)))

    model.add(keras.layers.Dropout(0.3, noise_shape=None, seed=None))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(keras.layers.Dense(64, activation="relu"))

    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

parser = parse_args()
print(parser)

x_train, y_train, x_test, y_test = load_data(parser)
NUM_CLASSES = 100 if parser.dataset=='cifar100' else 10
shadow_result_path = Path(parser.log_path) / 'attack' / f"{parser.model}-{parser.dataset}" \
                          / f"num_shadows{parser.num_shadows}-shadow_epoch{parser.shadow_epoch}-shadow_dataset_size{parser.shadow_dataset_size}"

def train_shadow():
    """ train the shadow model using real test data, returns the attack training set"""
    target_model_empty = load_model(parser)
    target_model_empty.compile(loss='categorical_crossentropy',
                               optimizer=keras.optimizers.Adam(learning_rate=parser.shadow_lr), metrics=['accuracy'])

    smb = ShadowModelBundle(
        target_model_empty,
        shadow_dataset_size=parser.shadow_dataset_size,
        model_path=shadow_result_path,
        num_models=parser.num_shadows,
        data_generator=datagen()
    )


    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        x_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)
    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(   ### NO data generator here
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=parser.shadow_epoch,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
        ),
    )
    return X_shadow, y_shadow


def train_attacker(X_shadow, y_shadow):
    """train the attacker using the attack training set"""
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES, class_one_hot_coded=True)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=parser.attack_epoch, verbose=True)
    )
    return amb


def test_attacker(model_path, amb):
    """test the attacker on real train/test set"""
    target_model = load_model(parser,'load',model_path)
    data_in = x_train[:parser.attack_test_dataset_size], y_train[:parser.attack_test_dataset_size]
    data_out = x_test[:parser.attack_test_dataset_size], y_test[:parser.attack_test_dataset_size]
    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )
    # Compute the attacker predict outputs
    probs = amb.predict_proba(attack_test_data)[:,1]
    metrics(probs,real_membership_labels, model_path)


def metrics(probs, ground_truth, model_path):
    predicts_5 = (probs > 0.5)
    acc_5, precision_5, recall_5, f1_5, advantage_5 = acc_pre_recall_adv(predicts_5, ground_truth)
    AP = average_precision_score(ground_truth,probs)
    AUC = roc_auc_score(ground_truth, probs)

    with open(Path(parser.log_path) / 'attack' / 'mia_results.txt', 'a+') as log:
        log.write('\n')
        log.write(model_path + f" num_shadows{parser.num_shadows}, shadow_epoch{parser.shadow_epoch}-shadow_dataset_size{parser.shadow_dataset_size}" + ":\n")
        log.write(f"Threshold 0.5 \n")
        log.write(
            "Acc:{:.5f} | Advantage:{:.5f} | Precision:{:.5f} | Recall:{:.5f} | F1:{:.5f}".format(acc_5, advantage_5 ,precision_5, recall_5, f1_5,
                                                                             ) + "\n")
        log.write("Average Precision: {:.5f} | AUC: {:.5f}\n".format(AP, AUC))

def acc_pre_recall_adv(predicts, ground_truth):
    accuracy = np.mean(predicts == ground_truth)

    true_positive = sum(predicts[:parser.attack_test_dataset_size])
    all_positive = sum(predicts)
    if all_positive == 0:
        precision = 0.
    else:
        precision = true_positive / all_positive
    recall = true_positive / parser.attack_test_dataset_size
    if precision == 0. and recall == 0.:
        f1 = 0.
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    advantage = np.abs(recall - (all_positive - true_positive) / parser.attack_test_dataset_size)

    return accuracy, precision, recall, f1, advantage


def main():
    X_shadow, y_shadow = train_shadow()
    if not shadow_result_path.exists():
        shadow_result_path.mkdir(parents=True)

    with open(shadow_result_path.joinpath('X_shadow'), 'wb') as xp:
        pickle.dump(X_shadow, xp)
    with open(shadow_result_path.joinpath('Y_shadow'), 'wb') as yp:
        pickle.dump(y_shadow, yp)
    amb = train_attacker(X_shadow, y_shadow)
    for model_path in parser.model_paths:
        test_attacker(model_path, amb)


def recover(path_shadow_predicts):
    """recover the attack training set from previously dumped file"""
    with open(os.path.join(path_shadow_predicts,'X_shadow'), 'rb') as xp:
        X_shadow = pickle.load(xp)
    with open(os.path.join(path_shadow_predicts,'Y_shadow'), 'rb') as yp:
        y_shadow = pickle.load(yp)
    amb = train_attacker(X_shadow, y_shadow)
    for model_path in parser.model_paths:
        test_attacker(model_path, amb)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpu

    # path_shadow_predicts = 'logs/attack/vgg-cifar10/num_shadows4-shadow_epoch30-shadow_dataset_size4500'
    # recover(path_shadow_predicts)

    main()
