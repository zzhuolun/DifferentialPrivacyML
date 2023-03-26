"""
Train the model with predefined training settings (such as batch_size, epochs, etc.)
and DP settings (such as l2_clip_norm, noise_multiplier, etc.).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

import sys
import os
from pathlib import Path

sys.path.append('../privai')

from privacy.privacy_monitoring.privacy_calculator.privacy_accountant import PrivacyAccountant
from privacy.privacy_monitoring.privacy_tracker import privacy_ledger
from privacy.dpml_algorithms.gradient_perturbation.dp_optimizer.dp_optimizer import DPAdamGaussianOptimizer as dp_opt
from privacy.dpml_algorithms.gradient_perturbation.dp_optimizer.dp_optimizer import \
    AdaDPAdamGaussianOptimizer as ada_dp_opt
from privacy.dpml_algorithms.gradient_perturbation.dp_optimizer.dp_optimizer import \
    QuantileAdaClipDPAdamGaussianOptimizer as quantile_adaclip_dp_opt
from utils import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dpsgd', type=str2bool, default=True, help='whether use DP')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model', type=str, default='mnist_model',
                        help='which model to train: vgg/lenet/vgg_overfit/tftu/mnist_model')
    parser.add_argument('--dataset', type=str, default='cifar10', help='mnist/cifar100/cifar10')
    parser.add_argument('--split_ratio', type=float, default=0.5, help='ratio of train/test')
    parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')

    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=250, help='Batch size')
    parser.add_argument('--log', type=str2bool, default=True, help='log the checkpoint and tensorboard or not')
    parser.add_argument('--log_path', type=str, default='logs', help='the directory to store all the results')
    parser.add_argument('--dataset_split_path', type=str, default=None, help='path to the file which stores the previously randomly split train/test data')

    # Privacy settings
    parser.add_argument('--noise_multiplier', type=float, default=1.0,
                        help='Ratio of the standard deviation to the clipping norm')
    parser.add_argument('--microbatches', type=int, default=250,
                        help='Number of microbatches (must evenly divide batch_size)')
    parser.add_argument('--l2_norm_clip', type=float, default=1.0, help='l2 clipping norm')
    parser.add_argument('--clip_mode', type=str, default='cclip',
                        help='"cclip":constant clip; "aclip": adaptive clip')
    parser.add_argument('--noise_decay_mode', type=str, default='nodecay',
                        help='nodecay:no decay; "step": step decay; "time": time decay')

    parser.add_argument('--expected_unclipped_quantile', type=float, default=0.9,
                        help='factor for updating adaptive clipping bound')
    parser.add_argument('--clip_update_rate', type=float, default=0.5,
                        help='factor for updating adaptive clipping bound')
    parser.add_argument('--clip_privacy_ratio', type=float, default=0.1,
                        help='factor for updating adaptive clipping bound')
    parser.add_argument('--target_delta', type=float, default=1e-6, help='Target delta for calculating privacy')
    parser.add_argument('--noise_decay_rate', type=float, default=0.00003,
                        help='decay rate in adaptive noise-based DPSGD')
    parser.add_argument('--ledger', type=str2bool, default=False, help='Record query history with PrivacyLedger')

    parser = parser.parse_args()
    parser.time_stamp = time_stamp()
    return parser


def opt_loss(FLAGS, pop_size):
    """
    define the optimizer and loss function for training
    Args:
        FLAGS: argument parser
        pop_size: population_size
    """
    if FLAGS.dpsgd:
        # Important: Compute vector of per-example loss rather than its mean over
        # a minibatch by setting the reduction=tf.losses.Reduction.NONE.
        loss = keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.losses.Reduction.NONE)

        if FLAGS.ledger:
            ledger = privacy_ledger.PrivacyLedger(population_size=pop_size,
                                                  selection_probability=(FLAGS.batch_size / pop_size))
        else:
            ledger = None

        if FLAGS.noise_decay_mode in ['step', 'time'] and FLAGS.clip_mode == 'cclip':
            print("train with adaptive noise mode")
            optimizer = ada_dp_opt(l2_norm_clip=FLAGS.l2_norm_clip,
                                   noise_multiplier=FLAGS.noise_multiplier,
                                   num_microbatches=FLAGS.microbatches,
                                   learning_rate=FLAGS.learning_rate,
                                   noise_decay_rate=FLAGS.noise_decay_rate,
                                   noise_decay_mode=FLAGS.noise_decay_mode,
                                   ledger=ledger)

        elif FLAGS.clip_mode == 'aclip':
            print("train with adaptive clip mode")
            optimizer = quantile_adaclip_dp_opt(l2_norm_clip=FLAGS.l2_norm_clip,
                                                noise_multiplier=FLAGS.noise_multiplier,
                                                noise_decay_rate=FLAGS.noise_decay_rate,
                                                noise_decay_mode=FLAGS.noise_decay_mode,
                                                expected_unclipped_quantile=FLAGS.expected_unclipped_quantile,
                                                clip_update_rate=FLAGS.clip_update_rate,
                                                clip_privacy_ratio=FLAGS.clip_privacy_ratio,
                                                num_microbatches=FLAGS.microbatches,
                                                learning_rate=FLAGS.learning_rate,
                                                ledger=ledger)

        elif FLAGS.noise_decay_mode == 'nodecay' and FLAGS.clip_mode == 'cclip':
            print("train with normal dpsgd mode")
            print("noise multiplier ", FLAGS.noise_multiplier)
            optimizer = dp_opt(l2_norm_clip=FLAGS.l2_norm_clip,
                               noise_multiplier=FLAGS.noise_multiplier,
                               num_microbatches=FLAGS.microbatches,
                               learning_rate=FLAGS.learning_rate,
                               ledger=ledger)
        else:
            raise ValueError('unsupported gradient clip and noise decay mode')

    else:
        print("train without dp")
        optimizer = keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
        loss = keras.losses.CategoricalCrossentropy(from_logits=False)
        ledger = None
    return optimizer, loss, ledger


def main(FLAGS):
    if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Number of microbatches should divide evenly batch_size')

    # Load training and test data.
    train_data, train_labels, test_data, test_labels = load_data(FLAGS)
    # Load model
    model = load_model(FLAGS)

    optimizer, loss, ledger = opt_loss(FLAGS, train_data.shape[0])

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # if data_augement:
    datagenerator = datagen()
    datagenerator.fit(train_data)

    model.fit_generator(datagenerator.flow(train_data, train_labels, batch_size=FLAGS.batch_size),
                        steps_per_epoch=train_data.shape[0] // FLAGS.batch_size,
                        epochs=FLAGS.epochs,
                        validation_data=(test_data, test_labels),
                        callbacks=my_callbacks(FLAGS, ckpt=FLAGS.log, tensorboard=FLAGS.log, lr=False), verbose=2
                        )

    if FLAGS.dpsgd:
        print('Start computing eps and alpha')
        total_iters = train_data.shape[0] * FLAGS.epochs // FLAGS.batch_size
        decaymode = None if FLAGS.noise_decay_mode == 'nodecay' else FLAGS.noise_decay_mode
        # use PrivacyAccountant to calculate the privacy cost, which works as a combination of moment accountant
        # and Fourier accountant
        if FLAGS.ledger:
            samples, queries = ledger.get_unformatted_ledger()
            ledger_sample_entries = privacy_ledger.format_ledger(K.get_value(samples),
                                                                 K.get_value(queries))

            pa = PrivacyAccountant(num_samples=train_data.shape[0], batch_size=FLAGS.batch_size,
                                   total_iters=total_iters, noise_multiplier=FLAGS.noise_multiplier,
                                   noise_decay_mode=decaymode,
                                   noise_decay_rate=FLAGS.noise_decay_rate,
                                   ledger=ledger_sample_entries)

            eps, delta = pa.compute_total_privacy(total_iters, target_delta=FLAGS.target_delta)

        else:
            pa = PrivacyAccountant(num_samples=train_data.shape[0], batch_size=FLAGS.batch_size,
                                   total_iters=total_iters, noise_multiplier=FLAGS.noise_multiplier,
                                   noise_decay_mode=decaymode,
                                   noise_decay_rate=FLAGS.noise_decay_rate)

            eps, delta = pa.compute_total_privacy(total_iters, target_delta=FLAGS.target_delta)

        print('Calculated total privacy spent: eps={:.4f}, delta={}'.format(eps, delta))
        save_path = logpath(FLAGS)[0] / f'eps{eps:.3f}-delta{delta}'
        model.save_weights(str(save_path))
        print('model saved at ' + str(save_path))
    else:
        save_path = logpath(FLAGS)[0] / 'nodp-final'
        model.save_weights(str(save_path))
        print('model saved at ' + str(logpath(FLAGS)[0]))


if __name__ == '__main__':
    FLAGS = parse_args()
    print(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    main(FLAGS)
