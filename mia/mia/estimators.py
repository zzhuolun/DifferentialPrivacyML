"""
Scikit-like estimators for the attack model and shadow models.
"""

import sklearn
from sklearn import metrics
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import os
from pathlib import Path
from collections import defaultdict


class ShadowModelBundle(sklearn.base.BaseEstimator):
    """
    A bundle of shadow models.

    :param model_fn: Function that builds a new shadow model
    :param shadow_dataset_size: Size of the training data for each shadow model
    :param num_models: Number of shadow models
    :param seed: Random seed
    :param ModelSerializer serializer: Serializer for the models. If None,
            the shadow models will be stored in memory. Otherwise, loaded
            and saved when needed.
    """

    MODEL_ID_FMT = "shadow_%d"

    def __init__(
        self, model_fn, shadow_dataset_size, model_path, num_models=20, seed=42, serializer=None, data_generator = None
    ):
        super().__init__()
        self.model_fn = model_fn
        self.shadow_dataset_size = shadow_dataset_size
        self.num_models = num_models
        self.seed = seed
        self.serializer = serializer
        self._reset_random_state()
        self.model_path = model_path
        self.data_generator = data_generator
    def fit_transform(self, X, y, verbose=False, fit_kwargs=None):
        """Train the shadow models and get a dataset for training the attack.

        :param X: Data coming from the same distribution as the target
                  training data
        :param y: Data labels
        :param bool verbose: Whether to display the progressbar
        :param dict fit_kwargs: Arguments that will be passed to the fit call for
                each shadow model.

        .. note::
            Be careful when holding out some of the passed data for validation
            (e.g., if using Keras, passing `fit_kwargs=dict(validation_split=0.7)`).
            Such data will be marked as "used in training", whereas it was used for
            validation. Doing so may decrease the success of the attack.
        """
        self._fit(X, y, verbose=verbose, fit_kwargs=fit_kwargs)
        return self._transform(verbose=verbose)

    def _reset_random_state(self):
        self._prng = np.random.RandomState(self.seed)

    def _get_model_iterator(self, indices=None, verbose=False):
        if indices is None:
            indices = range(self.num_models)
        if verbose:
            indices = tqdm(indices)
        return indices

    def _get_model(self, model_index):
        if self.serializer is not None:
            model_id = ShadowModelBundle.MODEL_ID_FMT % model_index
            model = self.serializer.load(model_id)
        else:
            model = self.shadow_models_[model_index]
        return model

    def _fit(self, X, y, verbose=False, pseudo=False, fit_kwargs=None):
        """Train the shadow models.

        .. note::
        Be careful not to hold out some of the passed data for validation
        (e.g., if using Keras, passing `fit_kwargs=dict(validation_split=0.7)`).
        Such data will be incorrectly marked as "used in training", whereas
        it was not.

        :param X: Data coming from the same distribution as the target
                  training data
        :param y: Data labels
        :param bool verbose: Whether to display the progressbar
        :param bool pseudo: If True, does not fit the models
        :param dict fit_kwargs: Arguments that will be passed to the fit call for
                each shadow model.
        """
        self.shadow_train_indices_ = []
        self.shadow_test_indices_ = []

        if self.serializer is None:
            self.shadow_models_ = []

        fit_kwargs = fit_kwargs or {}
        indices = np.arange(X.shape[0])

        lr_drop = 20

        def lr_scheduler(epoch):
            return (5e-4) * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        for i in self._get_model_iterator(verbose=verbose):
            # Pick indices for this shadow model.
            # log_dir = os.path.join('logs/shadow' , os.path.basename(self.model_path),f'{i}')
            tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(self.model_path / f'shadow{i}'), histogram_freq=1)
            shadow_indices = self._prng.choice(
                indices, 2 * self.shadow_dataset_size, replace=False
            )
            train_indices = shadow_indices[: self.shadow_dataset_size]
            test_indices = shadow_indices[self.shadow_dataset_size :]
            X_train, y_train = X[train_indices], y[train_indices]
            self.shadow_train_indices_.append(train_indices)
            self.shadow_test_indices_.append(test_indices)

            if pseudo:
                continue

            # Train the shadow model.
            shadow_model = self.model_fn
            if self.data_generator:
                self.data_generator.fit(X_train)
                shadow_model.fit_generator(self.data_generator.flow(X_train,y_train),callbacks= [reduce_lr, tensorboard_callback], **fit_kwargs)
            else:
                shadow_model.fit(X_train, y_train, callbacks= [reduce_lr, tensorboard_callback], **fit_kwargs)
            if self.serializer is not None:
                self.serializer.save(ShadowModelBundle.MODEL_ID_FMT % i, shadow_model)
            else:
                self.shadow_models_.append(shadow_model)

        self.X_fit_ = X
        self.y_fit_ = y
        self._reset_random_state()
        return self

    def _pseudo_fit(self, X, y, verbose=False, fit_kwargs=None):
        self._fit(X, y, verbose=verbose, fit_kwargs=fit_kwargs, pseudo=True)

    def _transform(self, shadow_indices=None, verbose=False):
        """Produce in/out data for training the attack model.

        :param shadow_indices: Indices of the shadow models to use
                for generating output data.
        :param verbose: Whether to show progress
        """
        shadow_data_array = []
        shadow_label_array = []

        model_index_iter = self._get_model_iterator(
            indices=shadow_indices, verbose=verbose
        )

        for i in model_index_iter:
            shadow_model = self._get_model(i)
            train_indices = self.shadow_train_indices_[i]
            test_indices = self.shadow_test_indices_[i]

            train_data = self.X_fit_[train_indices], self.y_fit_[train_indices]
            test_data = self.X_fit_[test_indices], self.y_fit_[test_indices]
            shadow_data, shadow_labels = prepare_attack_data(
                shadow_model, train_data, test_data
            )

            shadow_data_array.append(shadow_data)
            shadow_label_array.append(shadow_labels)

        X_transformed = np.vstack(shadow_data_array).astype("float32")
        y_transformed = np.hstack(shadow_label_array).astype("float32")
        return X_transformed, y_transformed


def prepare_attack_data(model, data_in, data_out):
    """
    Prepare the data in the attack model format.

    :param model: Classifier
    :param (X, y) data_in: Data used for training
    :param (X, y) data_out: Data not used for training

    :returns: (data, labels) for the attack classifier
    data.shape = (X_in.shape[0] + X_out.shape[0], 2*class_num)
    labels: binary array indicating whether the data in in or out training set
    """
    X_in, y_in = data_in
    X_out, y_out = data_out
    y_hat_in = model.predict(X_in)
    y_hat_out = model.predict(X_out)
    labels = np.ones(y_in.shape[0])
    labels = np.hstack([labels, np.zeros(y_out.shape[0])])    #[1,1,...,1,0,0,...,0]
    # TODO: this does not work for non-one-hot labels.
    # data = np.hstack([y_hat_in, y_in])
    data = np.c_[y_hat_in, y_in]
    data = np.vstack([data, np.c_[y_hat_out, y_out]])
    return data, labels


class AttackModelBundle(sklearn.base.BaseEstimator):
    """
    A bundle of attack models, one for each target model class.

    :param model_fn: Function that builds a new shadow model
    :param num_classes: Number of classes
    :param ModelSerializer serializer: Serializer for the models. If not None,
            the models will not be stored in memory, but rather loaded
            and saved when needed.
    :param class_one_hot_encoded: Whether the shadow data uses one-hot encoded
            class labels.
    """

    MODEL_ID_FMT = "attack_%d"

    def __init__(
        self, model_fn, num_classes, serializer=None, class_one_hot_coded=True
    ):
        self.model_fn = model_fn
        self.num_classes = num_classes
        self.serializer = serializer
        self.class_one_hot_coded = class_one_hot_coded

    def fit(self, X, y, verbose=False, fit_kwargs=None):
        """Train the attack models.

        :param X: Shadow predictions coming from
                  :py:func:`ShadowBundle.fit_transform`.
        :param y: Ditto
        :param verbose: Whether to display the progressbar
        :param fit_kwargs: Arguments that will be passed to the fit call for
                each attack model.
        """
        X_total = X[:, : self.num_classes]
        classes = X[:, self.num_classes :]

        datasets_by_class = []
        data_indices = np.arange(X_total.shape[0])
        for i in range(self.num_classes):
            if self.class_one_hot_coded:
                class_indices = data_indices[np.argmax(classes, axis=1) == i]
            else:
                class_indices = data_indices[np.squeeze(classes) == i]

            datasets_by_class.append((X_total[class_indices], y[class_indices]))

        if self.serializer is None:
            self.attack_models_ = []

        dataset_iter = datasets_by_class
        if verbose:
            dataset_iter = tqdm(dataset_iter)
        for i, (X_train, y_train) in enumerate(dataset_iter):  # train an attacker for each class
            print(f'Training attacker for class {i}')
            model = self.model_fn()
            fit_kwargs = fit_kwargs or {}
            model.fit(X_train, y_train, **fit_kwargs)

            if self.serializer is not None:
                model_id = AttackModelBundle.MODEL_ID_FMT % i
                self.serializer.save(model_id, model)
            else:
                self.attack_models_.append(model)

    def _get_model(self, model_index):
        if self.serializer is not None:
            model_id = AttackModelBundle.MODEL_ID_FMT % model_index
            model = self.serializer.load(model_id)
        else:
            model = self.attack_models_[model_index]
        return model

    def predict_proba(self, X):
        result = np.zeros((X.shape[0], 2))
        shadow_preds = X[:, : self.num_classes]
        classes = X[:, self.num_classes :]
        data_indices = np.arange(shadow_preds.shape[0])
        for i in range(self.num_classes):
            model = self._get_model(i)
            if self.class_one_hot_coded:
                class_indices = data_indices[np.argmax(classes, axis=1) == i]
            else:
                class_indices = data_indices[np.squeeze(classes) == i]
            membership_preds = model.predict(shadow_preds[class_indices])
            for j, example_index in enumerate(class_indices):
                prob = np.squeeze(membership_preds[j])
                result[example_index, 1] = prob
                result[example_index, 0] = 1 - prob
        return result

    def predict(self, X, real_membership_labels):
        classes = X[:, : self.num_classes]
        classes = np.argmax(classes, axis=1)
        probs = self.predict_proba(X)[:, 1]
        data_indices = np.arange(shadow_preds.shape[0])

        metric = defaultdict(float)
        metric['threshold'] = []
        for i in range(self.num_classes):
            class_indices = data_indices[classes == i]
            prob = probs[class_indices]
            label = real_membership_labels[class_indices]
            metric['AP'] += metrics.average_precision_score(label, prob)
            merric['AUC'] += metrics.roc_auc_score(label, prob)
            metric['acc_5'] += metrics.accuracy_score(label, prob>0.5)
            metric['precision_5'] += metrics.precision_score(label, prob>0.5)
            metric['recall_5'] += metrics.recall_score(label, prob>0.5)
            metric['f1_5'] += metrics.f1_score(label, prob>0.5)
            metric['adv_5'] += advantage(label, prob>0.5)
            thresholds = np.arange(0.1,1.0,0.1)
            accuracies = [metrics.accuracy_score(label, prob>thres) for thres in thresholds]
            thres_max_acc = thresholds[np.argmax(np.asarray(accuracies))]
            metric['threshold'].append(thres_max_acc)
            metric['acc_best'] += metrics.accuracy_score(label, prob > thres_max_acc)
            metric['precision_best'] += metrics.precision_score(label, prob > thres_max_acc)
            metric['recall_best'] += metrics.recall_score(label, prob > thres_max_acc)
            metric['f1_best'] += metrics.f1_score(label, prob > thres_max_acc)
            metric['adv_best'] += advantage(label, prob > thres_max_acc)
        for key in metric:
            if key is not 'threshold':
                metric[key]/=self.num_classes

        return metric
        # predictions=[]
        # thresholds = np.arange(0.1,1.0,0.1)
        # for thres in thresholds:
        #     predictions.append(probs>thres)
        # return predictions, thresholds

def advantage(y_true, y_pred):
    tp = np.sum(y_pred[y_true==1])
    fp = np.sum(y_pred) - tp

    tpr = tp / np.sum(y_true)
    fpr = fp / (y_true.shape[0] - np.sum(y_true))
    return np.abs(tpr - fpr)