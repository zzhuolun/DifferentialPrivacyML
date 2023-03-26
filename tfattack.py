"""
This script runs the Tensorflow MIA against a pretrained model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import tensorflow as tf

sys.path.append('../privacy/')

import os
from tensorflow.keras.datasets import cifar100
from tensorflow import keras
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackInputData
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import get_averaged_attack_metrics
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import SlicingSpec
from tensorflow_privacy.privacy.membership_inference_attack.data_structures import AttackType
from tensorflow_privacy.privacy.membership_inference_attack import membership_inference_attack as mia
import tensorflow_privacy.privacy.membership_inference_attack.plotting as plotting
import pickle
import argparse
from matplotlib import pyplot as plt
from utils import *
import time
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_paths', nargs='+', help='the paths of vgg models to be attacked, can be multiple paths')

    parser.add_argument('--model', type=str, default='mnist_model', help='vgg/lenet/mnist_model')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist/cifar100')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('--dataset_split_path', type=str, default=None,
                        help='path to the file which stores the previously randomly split train/test data')
    parser.add_argument('--log_path', type=str, default='logs', help='the directory to store all the results')
    args = parser.parse_args()
    return args


parser = parse_args()
print(parser)
NUM_CLASSES = 100 if parser.dataset == 'cifar100' else 10


def get_predictions(model_path, logits=False):
    """
    get predictions of the target model on the real training/test set
    Args:
        model_path: the path to the target model
        logits: if True, the last layer activation of the model will be changed to linear activation and the model
                will output logits

    Returns:
        x_train: real train data
        y_train: real train label
        x_test: real test data
        y_test: real tes label
        predicts_train: outputs of target model on train data
        predicts_test: outputs of target model on test data
    """
    x_train, y_train, x_test, y_test = load_data(parser)
    model = load_model(parser, 'load', model_path)
    if logits:
        model = probs_to_logits(model)
    predicts_train = model.predict(x_train)
    predicts_test = model.predict(x_test)
    return x_train, y_train, x_test, y_test, predicts_train, predicts_test


def main(model_path, attack_by_class=True, attack_entiredataset=False):
    """
    run TF MIA on the target model
    Args:
        model_path: target model path
        attack_by_class: if True, TF MIA will train an attacker for each class of the dataset
        attack_entiredataset: if True, TF MIA will train on attacker for the entire dataset

    Returns:
        attack_result(class AttackResults): the attack result
    """""
    x_train, y_train, x_test, y_test, probs_train, probs_test = get_predictions(model_path)
    labels_train = np.argmax(probs_train, 1)
    labels_test = np.argmax(probs_test, 1)
    train_acc = sum(labels_train == np.argmax(y_train, 1)) / y_train.shape[0]
    print('---------------acc----------------')
    print("the train accuracy is: ", train_acc)
    test_acc = sum(labels_test == np.argmax(y_test, 1)) / y_test.shape[0]
    print("the test accuracy is: ", test_acc)
    print('---------------------------------')
    # y_train_onehot = keras.utils.to_categorical(y_train, 100)
    # y_test_onehot = keras.utils.to_categorical(y_test, 100)
    loss_train = keras.losses.categorical_crossentropy(
        y_train, probs_train, from_logits=False, label_smoothing=0)
    loss_test = keras.losses.categorical_crossentropy(
        y_test, probs_test, from_logits=False, label_smoothing=0)
    # loss_train = loss_train.numpy()
    # loss_test = loss_test.numpy()
    loss_train = loss_train.eval(session=tf.Session())
    loss_test = loss_test.eval(session=tf.Session())
    input = AttackInputData(
        probs_train=probs_train,
        probs_test=probs_test,
        loss_train=loss_train,
        loss_test=loss_test,
        labels_train=np.argmax(y_train, 1),  # groundtruth classes
        labels_test=np.argmax(y_test, 1)
    )

    attacks_result = mia.run_attacks(input,
                                     SlicingSpec(
                                         entire_dataset=attack_entiredataset,
                                         by_class=attack_by_class,
                                         # by_classification_correctness = True
                                     ),
                                     attack_types=[
                                         AttackType.META_ATTACK,
                                         # AttackType.THRESHOLD_ATTACK,
                                         # AttackType.LOGISTIC_REGRESSION,
                                         # AttackType.MULTI_LAYERED_PERCEPTRON,
                                         # AttackType.RANDOM_FOREST,
                                         # AttackType.K_NEAREST_NEIGHBORS,
                                         # AttackType.SVM_ATTACK,
                                     ])

    # Print a user-friendly summary of the attacks
    print(attacks_result.summary(by_slices=True))
    return attacks_result


def _log(attacks_outputs):
    """
    Write the attack results to tf_results.txt
    Args:
        attacks_outputs: the attack results
        identifier:
    """
    with open(Path(parser.log_file) / 'attack' / 'tf_results.txt', 'a+') as log:
        log.write('\n')
        for attacks_result, model_path in attacks_outputs:
            log.write(model_path + ":\n")
            metric_dict, metric_avg_dict, metric_avg_all, summary = get_averaged_attack_metrics(attacks_result)
            log.write(summary)
            log.write('Average attacker performance: \n')
            log.write(
                f"Accuracy: {round(metric_avg_all['acc'], 4)} | Advantage: {round(metric_avg_all['adv'], 4)} " +
                f"| Precision: {round(metric_avg_all['precision'], 4)} | AP: {round(metric_avg_all['AP'], 4)} " +
                f"| AUC: {round(metric_avg_all['auc'], 4)} | Recall: {round(metric_avg_all['recall'], 4)} | F1: {round(metric_avg_all['f1'], 4)}\n")
            for k in metric_avg_dict:
                log.write(
                    f"{k:17s}: Accuracy: {round(metric_avg_dict[k]['acc'], 4)} | Advantage: {round(metric_avg_dict[k]['adv'], 4)} " +
                    f"| Precision: {round(metric_avg_dict[k]['precision'], 4)} | AP: {round(metric_avg_dict[k]['AP'], 4)} " +
                    f"| AUC: {round(metric_avg_dict[k]['auc'], 4)} | Recall: {round(metric_avg_dict[k]['recall'], 4)} | F1: {round(metric_avg_dict[k]['f1'], 4)}\n")
            log.write('\n')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = parser.gpu
    metric_dicts = {}
    log_input = []
    for model_path in parser.model_paths:
        attack_result = main(model_path)
        metric_dicts[model_path.split('/')[-1]] = attack_result
        log_input.append(tuple((attack_result, model_path)))

    _log(log_input)

