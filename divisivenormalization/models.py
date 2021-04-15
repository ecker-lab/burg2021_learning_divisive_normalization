import inspect
import os
import random
from typing import Tuple
import numpy as np
import tensorflow as tf
from scipy import stats
from tensorflow.contrib import layers

from divisivenormalization.regularizers import (
    smoothness_regularizer_2d,
    group_sparsity_regularizer_2d,
    smoothness_regularizer_1d,
)


def inv_elu(x):
    """Inverse elu function."""
    y = x.copy()
    idx = y < 1.0
    y[idx] = np.log(y[idx]) + 1.0
    return y


def lin_step(x, a, b):
    return tf.minimum(tf.constant(b - a, dtype=tf.float32), tf.nn.relu(x - tf.constant(a, dtype=tf.float32))) / (b - a)


def tent(x, a, b):
    z = tf.constant(0, dtype=tf.float32)
    d = tf.constant(2 * (b - a), dtype=tf.float32)
    a = tf.constant(a, dtype=tf.float32)
    return tf.minimum(tf.maximum(x - a, z), tf.maximum(a + d - x, z)) / (b - a)


def output_nonlinearity(x, num_neurons, vmin=-3.0, vmax=6.0, num_bins=10, alpha=0, scope="output_nonlinearity"):
    with tf.variable_scope(scope):
        elu = tf.nn.elu(x - 1.0) + 1.0
        if alpha == -1:
            tf.add_to_collection("output_nonlinearity", 0)
            return elu
        _, neurons = x.get_shape().as_list()
        k = int(num_bins / 2)
        num_bins = 2 * k
        bins = np.linspace(vmin, vmax, num_bins + 1, endpoint=True)
        segments = [tent(x, a, b) for a, b in zip(bins[:-2], bins[1:-1])] + [lin_step(x, bins[-2], bins[-1])]
        reg = lambda w: smoothness_regularizer_1d(w, weight=alpha, order=2)
        a = tf.get_variable(
            "weights",
            shape=[neurons, num_bins, 1],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0),
            regularizer=reg,
        )
        a = tf.exp(a)
        tf.add_to_collection("output_nonlinearity", a)
        v = tf.transpose(tf.concat([tf.reshape(s, [-1, neurons, 1]) for s in segments], axis=2), [1, 0, 2])
        multiplier = tf.transpose(tf.reshape(tf.matmul(v, a), [neurons, -1]))
        return multiplier * elu


class Net:
    """Abstract class to be inherited by models."""

    def __init__(
        self, data=None, log_dir=None, log_hash=None, global_step=None, obs_noise_model="poisson", eval_batches=None
    ):
        self.data = data
        log_dir_ = os.path.dirname(os.path.dirname(inspect.stack()[0][1]))
        self.log_dir_wo_hash = os.path.join(log_dir_, "train_logs", "tmp") if log_dir is None else log_dir
        if log_hash == None:
            log_hash = "%010x" % random.getrandbits(40)
        self.log_dir = os.path.join(self.log_dir_wo_hash, log_hash)
        self.log_hash = log_hash
        self.global_step = 0 if global_step == None else global_step
        self.session = None
        self.obs_noise_model = obs_noise_model
        self.best_loss = 1e100
        self.val_iter_loss = []
        self.eval_batches = eval_batches

        # placeholders
        if data is None:
            return
        with tf.Graph().as_default() as self.graph:
            self.is_training = tf.placeholder(tf.bool)
            self.learning_rate = tf.placeholder(tf.float32)
            self.images = tf.placeholder(tf.float32, shape=[None, data.px_y, data.px_x, 1])
            self.responses = tf.placeholder(tf.float32, shape=[None, data.num_neurons])
            self.real_responses = tf.placeholder(tf.float32, shape=[None, data.num_neurons])

    def initialize(self):
        self.summaries = tf.summary.merge_all()
        if self.session is None:
            self.session = tf.Session(graph=self.graph)
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=100)
        self.saver_best = tf.train.Saver(max_to_keep=1)
        self.writer = tf.summary.FileWriter(self.log_dir, max_queue=0, flush_secs=0.1)

    def __del__(self):
        try:
            if not self.session == None:
                self.session.close()
                self.writer.close()
        except:
            pass

    def close(self):
        self.session.close()

    def save(self, step=None):
        if step == None:
            step = self.global_step
        chkp_file = os.path.join(self.log_dir, "model.ckpt")
        self.saver.save(self.session, chkp_file, global_step=step)

    def save_best(self):
        self.saver_best.save(self.session, os.path.join(self.log_dir, "best.ckpt"))

    def load(self, log_hash=None, omit_var_by_name=None):
        """Load model.

        Args:
            log_hash (str, optional): Checkpoint hash. Defaults to None.
            omit_var_by_name (list[str], optional): Variables that should not be loaded. Defaults to
                None. Example: ["conv0/weights"]
        """

        if log_hash is None:
            print("WARNING: Restored same model. (specified log hash to load from was None)")
            ckpt_path = os.path.join(self.log_dir, "model.ckpt")
        else:
            ckpt_path = os.path.join(self.log_dir_wo_hash, log_hash, "model.ckpt")

        ckpt_var_list = tf.train.list_variables(ckpt_path)
        var_list = []
        for v in ckpt_var_list:
            if omit_var_by_name is not None and v[0] in omit_var_by_name:
                continue
            var_list.append(self.graph.get_tensor_by_name(v[0] + ":0"))

        self.trainable_var_saver = tf.train.Saver(var_list=var_list)
        self.trainable_var_saver.restore(self.session, ckpt_path)

    def load_best(self):
        ckpt_path = os.path.join(self.log_dir, "best.ckpt")
        ckpt_var_list = tf.train.list_variables(ckpt_path)
        var_list = []
        for v in ckpt_var_list:
            var_list.append(self.graph.get_tensor_by_name(v[0] + ":0"))
            print("load", v[0] + ":0")

        self.saver_best = tf.train.Saver(var_list=var_list)
        self.saver_best.restore(self.session, ckpt_path)

    def train(
        self,
        max_iter=5000,
        learning_rate=3e-4,
        batch_size=256,
        val_steps=100,
        save_steps=1000,
        early_stopping_steps=5,
        learning_rule_updates=3,
        eval_batches=None,
    ):

        self.eval_batches = eval_batches
        with self.graph.as_default():
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            imgs_val, res_val, real_resp_val = self.data.val()
            not_improved = 0
            num_lr_updates = 0
            for i in range(self.global_step + 1, self.global_step + max_iter + 1):

                # training step
                imgs_batch, res_batch, real_batch = self.data.minibatch(batch_size)
                self.global_step = i
                feed_dict = {
                    self.images: imgs_batch,
                    self.responses: res_batch,
                    self.real_responses: real_batch,
                    self.is_training: True,
                    self.learning_rate: learning_rate,
                }
                self.session.run([self.train_step, update_ops], feed_dict)
                # validate/save periodically
                if not i % save_steps:
                    self.save(i)
                if not i % val_steps:
                    result = self.eval(
                        images=imgs_val,
                        responses=res_val,
                        real_responses=real_resp_val,
                        with_summaries=False,
                        global_step=i,
                        learning_rate=learning_rate,
                    )
                    if result[0] < self.best_loss:
                        self.best_loss = result[0]
                        self.save_best()
                        not_improved = 0
                    else:
                        not_improved += 1
                    if not_improved == early_stopping_steps:
                        self.global_step -= early_stopping_steps * val_steps
                        self.load_best()
                        not_improved = 0
                        learning_rate /= 3
                        print("reducing learning rate to {}".format(learning_rate))
                        num_lr_updates += 1
                        if num_lr_updates == learning_rule_updates:
                            self.load_best()
                            break
                    yield (i, result)

    def eval(
        self,
        with_summaries=False,
        keep_record_loss=True,
        images=None,
        responses=None,
        real_responses=None,
        global_step=None,
        learning_rate=None,
    ):
        """Returns result, where last entry are the predictions."""

        if (images is None) or (responses is None):
            images, responses, real_responses = self.data.test()
            nrep, nim, nneu = responses.shape
            images = np.tile(images, [nrep, 1, 1, 1])
            responses = responses.reshape([nim * nrep, nneu])
            real_responses = real_responses.reshape([nim * nrep, nneu])

        if with_summaries:
            raise NotImplementedError

        ops = self.get_test_ops()

        if self.eval_batches is not None:
            batches = self.eval_batches
            numpts = images.shape[0]
            numbatch = int(np.ceil(numpts / batches))
            pred_val = []
            result = 0
            for batch_idx in range(0, numbatch):
                if batches * (batch_idx + 1) > numpts:
                    idx = (batch_idx * batches) + np.arange(0, numpts - (batch_idx * batches))
                else:
                    idx = (batch_idx * batches) + np.arange(0, batches)

                feed_dict = {
                    self.images: images[idx],
                    self.responses: responses[idx],
                    self.real_responses: real_responses[idx],
                    self.is_training: False,
                }
                res = self.session.run(ops, feed_dict)
                pred_val.append(res[-1])
                result += np.array([r * len(idx) for r in res[:-1]])

            result = [np.float32(r / numpts) for r in result]
            pred_val = np.concatenate(pred_val)
            result.append(pred_val)

        else:
            feed_dict = {
                self.images: images,
                self.responses: responses,
                self.real_responses: real_responses,
                self.is_training: False,
            }
            result = self.session.run(ops, feed_dict)

        if keep_record_loss:
            self.val_iter_loss.append(result[0])
        return result

    def compute_log_likelihoods(self, prediction, response, real_responses):
        self.poisson = tf.reduce_mean(
            tf.reduce_sum((prediction - response * tf.log(prediction + 1e-9)) * real_responses, axis=0)
            / tf.reduce_sum(real_responses, axis=0)
        )

    def get_log_likelihood(self):
        if self.obs_noise_model == "poisson":
            return self.poisson
        else:
            raise NotImplementedError

    def get_test_ops(self):
        return [self.get_log_likelihood(), self.total_loss, self.prediction]

    def evaluate_corr_vals(self):
        """Computes and returns a vector of correlations between prediction and labels of all neurons
        on the validation set."""

        im, res, real_res = self.data.val()
        result = self.eval(images=im, responses=res, real_responses=real_res)
        pred = result[-1]

        corrs = []
        for i in range(self.data.num_neurons):
            # keep only entries corresponding to real_res
            r = res[:, i]
            p = pred[:, i]
            b = real_res[:, i].astype(np.bool)
            r = np.compress(b, r)
            p = np.compress(b, p)
            corr = stats.pearsonr(r, p)[0]
            if np.isnan(corr):
                print("INFO: corr for neuron " + str(i) + " is nan - replaced by 0")
                corr = 0
            corrs.append(corr)

        return corrs

    def evaluate_avg_corr_val(self):
        """Prediction correlation averaged across neurons on validation set."""

        avg_corr = np.mean(self.evaluate_corr_vals())
        return avg_corr

    def evaluate_ve_testset_per_neuron(self):
        """Computes variance explained and explainable variance on the test set per neuron."""

        images_test, responses_test, real_responses_test = self.data.test()
        nrep, nim, nneu = responses_test.shape
        predictions_test = self.prediction.eval(
            session=self.session, feed_dict={self.images: images_test, self.is_training: False}
        )
        predictions_test = np.tile(predictions_test.T, 4).T
        resps_test_nan = self.data.nanarray(real_responses_test, responses_test)

        MSE = np.nanmean((predictions_test - resps_test_nan.reshape([nrep * nim, nneu])) ** 2, axis=0)

        obs_var_avg, total_variance, explainable_var = [], [], []
        for n in range(nneu):
            rep = self.data.repetitions[n]
            resp_ = resps_test_nan[:rep, :, n]

            obs_var = np.nanmean((np.nanvar(resp_, axis=0, ddof=1)), axis=0)
            obs_var_avg.append(obs_var)

            tot_var = np.nanvar(resp_, axis=(0, 1), ddof=1)
            total_variance.append(tot_var)
            explainable_var.append(tot_var - obs_var)

        total_variance = np.array(total_variance)
        explainable_var = np.array(explainable_var)
        var_explained = total_variance - MSE

        return var_explained, explainable_var

    def evaluate_fev_testset_per_neuron(self):
        """Computes fraction of explainable variance explained on the test set per neuron."""

        var_explained, explainable_var = self.evaluate_ve_testset_per_neuron()
        eve = var_explained / explainable_var
        return eve

    def evaluate_fev_testset(self):
        """Computes average fraction of explainable variance explained on the test set."""
        return self.evaluate_fev_testset_per_neuron().mean()

    def show_tf_trainable_variables(self):
        """Print all trainable variables in the model graph."""
        with self.graph.as_default():
            for v in tf.trainable_variables():
                print(v)


class ConvSubunitNetOutputNonlin(Net):
    """Subunit model"""
    def __init__(
        self,
        filter_sizes,
        out_channels,
        strides,
        paddings,
        smooth_weights,
        sparse_weights,
        readout_sparse_weight,
        output_nonlin_smooth_weight,
        data=None,
        log_dir=None,
        log_hash=None,
        global_step=None,
        obs_noise_model="poisson",
        eval_batches=None,
    ):

        super().__init__(
            data=data,
            log_dir=log_dir,
            log_hash=log_hash,
            global_step=global_step,
            obs_noise_model=obs_noise_model,
            eval_batches=eval_batches,
        )
        self.conv = []
        self.W = []
        self.readout_sparseness_regularizer = 0.0

        with self.graph.as_default():
            # convolutional layers
            for i, (filter_size, out_chans, stride, padding, smooth_weight, sparse_weight) in enumerate(
                zip(filter_sizes, out_channels, strides, paddings, smooth_weights, sparse_weights)
            ):
                x = self.images if not i else self.conv[i - 1]
                bn_params = {"decay": 0.9, "is_training": self.is_training, "scale": False}
                scope = "conv{}".format(i)
                reg = lambda w: smoothness_regularizer_2d(w, smooth_weight) + group_sparsity_regularizer_2d(
                    w, sparse_weight
                )
                c = layers.convolution2d(
                    inputs=x,
                    num_outputs=out_chans,
                    kernel_size=int(filter_size),
                    stride=int(stride),
                    padding=padding,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=bn_params,
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                    weights_regularizer=reg,
                    scope=scope,
                )
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable("weights")
                self.W.append(W)
                self.conv.append(c)

            # Readout
            sz = self.conv[-1].get_shape()
            px_x_conv = int(sz[1])
            px_y_conv = int(sz[2])
            px_conv = px_x_conv * px_y_conv
            conv_flat = tf.reshape(self.conv[-1], [-1, px_conv, out_channels[-1], 1])
            self.W_spatial = tf.abs(
                tf.get_variable(
                    "W_spatial",
                    trainable=True,
                    shape=[px_conv, self.data.num_neurons],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                )
            )
            W_spatial_flat = tf.reshape(self.W_spatial, [px_conv, 1, 1, self.data.num_neurons])
            self.h_spatial = tf.nn.conv2d(
                conv_flat, W_spatial_flat, strides=[1, 1, 1, 1], padding="VALID"
            )  # Dot product

            self.W_features = tf.abs(
                tf.get_variable(
                    "W_features",
                    trainable=True,
                    shape=[out_channels[-1], self.data.num_neurons],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                )
            )
            self.h_features = self.h_spatial * self.W_features  # element wise product
            self.h_out = tf.reduce_sum(self.h_features, [1, 2])  # shape: (batch, 166)

            # L1 regularization for readout layer
            self.readout_sparseness_regularizer = readout_sparse_weight * tf.reduce_sum(
                tf.reduce_sum(tf.abs(self.W_spatial), 0) * tf.reduce_sum(tf.abs(self.W_features), 0)
            )
            tf.losses.add_loss(self.readout_sparseness_regularizer, tf.GraphKeys.REGULARIZATION_LOSSES)

            # output nonlinearity
            _, responses, _ = self.data.train()
            b = inv_elu(responses.mean(axis=0))
            self.b_out = tf.get_variable(
                "b_out", shape=[self.data.num_neurons], dtype=tf.float32, initializer=tf.constant_initializer(b)
            )
            self.prediction = tf.identity(
                output_nonlinearity(
                    self.h_out + self.b_out,
                    self.data.num_neurons,
                    vmin=-3,
                    vmax=6,
                    num_bins=50,
                    alpha=output_nonlin_smooth_weight,
                ),
                name="predictions",
            )

            # loss
            self.compute_log_likelihoods(self.prediction, self.responses, self.real_responses)
            tf.losses.add_loss(self.get_log_likelihood())
            self.total_loss = tf.losses.get_total_loss()

            # regularizers
            self.smoothness_regularizer = tf.add_n(tf.get_collection("smoothness_regularizer_2d"))
            self.group_sparsity_regularizer = tf.add_n(tf.get_collection("group_sparsity_regularizer_2d"))

            # optimizer
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            # initialize TF session
            self.initialize()

    def get_test_ops(self):
        return [
            self.get_log_likelihood(),
            self.readout_sparseness_regularizer,
            self.group_sparsity_regularizer,
            self.smoothness_regularizer,
            self.total_loss,
            self.prediction,
        ]


class ConvNet(Net):
    """Convolutional neural network"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = []
        self.W = []
        self.readout_sparseness_regularizer = 0.0

    def build(
        self,
        filter_sizes,
        out_channels,
        strides,
        paddings,
        smooth_weights,
        sparse_weights,
        readout_sparse_weight,
        output_nonlin_smooth_weight,
    ):

        with self.graph.as_default():
            # convolutional layers
            for i, (filter_size, out_chans, stride, padding, smooth_weight, sparse_weight) in enumerate(
                zip(filter_sizes, out_channels, strides, paddings, smooth_weights, sparse_weights)
            ):
                x = self.images if not i else self.conv[i - 1]
                bn_params = {"decay": 0.9, "is_training": self.is_training}
                scope = "conv{}".format(i)
                reg = lambda w: smoothness_regularizer_2d(w, smooth_weight) + group_sparsity_regularizer_2d(
                    w, sparse_weight
                )
                c = layers.convolution2d(
                    inputs=x,
                    num_outputs=out_chans,
                    kernel_size=int(filter_size),
                    stride=int(stride),
                    padding=padding,
                    activation_fn=tf.nn.elu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=bn_params,
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                    weights_regularizer=reg,
                    scope=scope,
                )
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable("weights")
                self.W.append(W)
                self.conv.append(c)

            # readout layer
            sz = c.get_shape()
            px_x_conv = int(sz[1])
            px_y_conv = int(sz[2])
            px_conv = px_x_conv * px_y_conv
            conv_flat = tf.reshape(c, [-1, px_conv, out_channels[-1], 1])
            self.W_spatial = tf.get_variable(
                "W_spatial",
                shape=[px_conv, self.data.num_neurons],
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            )
            W_spatial_flat = tf.reshape(self.W_spatial, [px_conv, 1, 1, self.data.num_neurons])
            h_spatial = tf.nn.conv2d(conv_flat, W_spatial_flat, strides=[1, 1, 1, 1], padding="VALID")  # dot product
            self.W_features = tf.get_variable(
                "W_features",
                shape=[out_channels[-1], self.data.num_neurons],
                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
            )
            self.h_out = tf.reduce_sum(tf.multiply(h_spatial, self.W_features), [1, 2])

            # L1 regularization for readout layer
            self.readout_sparseness_regularizer = readout_sparse_weight * tf.reduce_sum(
                tf.reduce_sum(tf.abs(self.W_spatial), 0) * tf.reduce_sum(tf.abs(self.W_features), 0)
            )
            tf.losses.add_loss(self.readout_sparseness_regularizer, tf.GraphKeys.REGULARIZATION_LOSSES)

            # output nonlinearity
            _, responses, _ = self.data.train()
            b = inv_elu(responses.mean(axis=0))
            self.b_out = tf.get_variable(
                "b_out", shape=[self.data.num_neurons], dtype=tf.float32, initializer=tf.constant_initializer(b)
            )
            self.prediction = output_nonlinearity(
                self.h_out + self.b_out,
                self.data.num_neurons,
                vmin=-3,
                vmax=6,
                num_bins=50,
                alpha=output_nonlin_smooth_weight,
            )

            self.output_regularizer = tf.add_n(tf.get_collection("smoothness_regularizer_1d"))

            # loss
            self.compute_log_likelihoods(self.prediction, self.responses, self.real_responses)
            tf.losses.add_loss(self.get_log_likelihood())
            self.total_loss = tf.losses.get_total_loss()

            # regularizers
            self.smoothness_regularizer = tf.add_n(tf.get_collection("smoothness_regularizer_2d"))
            self.group_sparsity_regularizer = tf.add_n(tf.get_collection("group_sparsity_regularizer_2d"))

            # optimizer
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            # initialize TF session
            self.initialize()

    def get_test_ops(self):
        return [
            self.get_log_likelihood(),
            self.readout_sparseness_regularizer,
            self.group_sparsity_regularizer,
            self.smoothness_regularizer,
            self.output_regularizer,
            self.total_loss,
            self.prediction,
        ]


class DivisiveNetOutputNonlin(Net):
    """Full divisive normalization model."""
    def __init__(
        self,
        filter_sizes,
        out_channels,
        strides,
        paddings,
        smooth_weights,
        sparse_weights,
        readout_sparse_weight,
        output_nonlin_smooth_weight,
        pool_kernel_size,
        pool_type,
        dilation,
        M,
        dn_u_size,
        abs_v_l1_weight,
        dn_padding="SAME",
        data=None,
        log_dir=None,
        log_hash=None,
        global_step=None,
        obs_noise_model="poisson",
        eval_batches=None,
    ):

        super().__init__(
            data=data,
            log_dir=log_dir,
            log_hash=log_hash,
            global_step=global_step,
            obs_noise_model=obs_noise_model,
            eval_batches=eval_batches,
        )
        self.conv = []
        self.W = []
        self.readout_sparseness_regularizer = 0.0

        with self.graph.as_default():
            # convolutional layers
            for i, (filter_size, out_chans, stride, padding, smooth_weight, sparse_weight) in enumerate(
                zip(filter_sizes, out_channels, strides, paddings, smooth_weights, sparse_weights)
            ):
                x = self.images if not i else self.conv[i - 1]
                bn_params = {"decay": 0.9, "is_training": self.is_training, "scale": False}
                scope = "conv{}".format(i)
                reg = lambda w: smoothness_regularizer_2d(w, smooth_weight) + group_sparsity_regularizer_2d(
                    w, sparse_weight
                )
                c = layers.convolution2d(
                    inputs=x,
                    num_outputs=out_chans,
                    kernel_size=int(filter_size),
                    stride=int(stride),
                    padding=padding,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=bn_params,
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                    weights_regularizer=reg,
                    scope=scope,
                )
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable("weights")
                self.W.append(W)
                self.conv.append(c)

            # Divisive normalization
            self.dn_exponent = tf.abs(
                tf.get_variable(
                    "dn_exponent",
                    shape=[1, 1, 1, self.conv[-1].shape[-1]],
                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.01),
                )
            )
            self.y_n = tf.pow(self.conv[-1], self.dn_exponent)

            if pool_type == "MAX":
                self.pooled = tf.nn.max_pool(self.y_n, pool_kernel_size, strides=[1, 1, 1, 1], padding="SAME")
            elif pool_type == "AVG":
                self.pooled = tf.nn.avg_pool(self.y_n, pool_kernel_size, strides=[1, 1, 1, 1], padding="SAME")
            else:
                raise NotImplementedError

            # get weights for factorized dilated convolution
            # u shape: (space, space, in-features to pool from, out-features being normalized, normalization pools m)
            u_shape = (dn_u_size[0], dn_u_size[1], 1, out_channels[-1], M)
            self.u_unnormalized = tf.Variable(
                initial_value=tf.random_normal(shape=u_shape, stddev=1e-3), trainable=True, name="u"
            )
            self.u = self.u_unnormalized / tf.norm(self.u_unnormalized, ord="fro", axis=[0, 1], keepdims=True)
            self.v = tf.Variable(
                initial_value=tf.random_normal(shape=(1, 1, out_channels[-1], out_channels[-1], M), stddev=1e-3),
                trainable=True,
                name="v",
            )
            self.abs_u = tf.abs(self.u)
            self.abs_v = tf.abs(self.v)

            # L1 regularizer on abs_v to enforce sparsity (use minimal set of features to normalize)
            self.abs_v_l1_penalty = abs_v_l1_weight * tf.reduce_sum(self.abs_v)
            tf.losses.add_loss(self.abs_v_l1_penalty, tf.GraphKeys.REGULARIZATION_LOSSES)

            self.p_m = self.abs_u * self.abs_v
            self.p = tf.reduce_sum(self.p_m, axis=-1)  # sum over normalization pools (m dimension)

            self.supp = tf.nn.conv2d(
                input=self.pooled, filter=self.p, strides=[1, 1, 1, 1], padding=dn_padding, dilations=dilation
            )

            self.semisaturation_const = tf.abs(
                tf.get_variable(
                    "semisaturation_const",
                    shape=[1, 1, 1, self.conv[-1].shape[-1]],
                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.01),
                )
            )

            self.abs_eta = tf.abs(self.semisaturation_const)
            self.eta_n = tf.pow(self.abs_eta, self.dn_exponent)

            denom = self.eta_n + self.supp

            # for valid padding in supp convolution: make sure, dimensions for the division fit
            if (dn_padding == "SAME") or (dn_u_size[0] == 1):  # no cropping required
                self.y_n_crop = self.y_n
            else:  # crop to fit self.supp size
                y_n_spatial_crop = (tf.shape(self.y_n)[1] - tf.shape(denom)[1]) // 2
                self.y_n_crop = self.y_n[:, y_n_spatial_crop:-y_n_spatial_crop, y_n_spatial_crop:-y_n_spatial_crop, :]

            denom_greater_zero = tf.Assert(
                tf.reduce_all(tf.greater(denom, 0.0)), [tf.reduce_min(denom), tf.reduce_max(denom)], name="denom"
            )
            with tf.control_dependencies([denom_greater_zero]):
                self.dn_res = self.y_n_crop / denom

            # Readout
            sz = self.dn_res.get_shape()
            px_x_conv = int(sz[1])
            px_y_conv = int(sz[2])
            px_conv = px_x_conv * px_y_conv
            conv_flat = tf.reshape(self.dn_res, [-1, px_conv, out_channels[-1], 1])
            self.W_spatial = tf.abs(
                tf.get_variable(
                    "W_spatial",
                    trainable=True,
                    shape=[px_conv, self.data.num_neurons],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                )
            )
            W_spatial_flat = tf.reshape(self.W_spatial, [px_conv, 1, 1, self.data.num_neurons])
            self.h_spatial = tf.nn.conv2d(
                conv_flat, W_spatial_flat, strides=[1, 1, 1, 1], padding="VALID"
            )  # dot product
            self.W_features = tf.abs(
                tf.get_variable(
                    "W_features",
                    trainable=True,
                    shape=[out_channels[-1], self.data.num_neurons],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                )
            )
            self.h_features = self.h_spatial * self.W_features  # element wise product
            self.h_out = tf.reduce_sum(self.h_features, [1, 2])

            # L1 regularization for readout layer
            self.readout_sparseness_regularizer = readout_sparse_weight * tf.reduce_sum(
                tf.reduce_sum(tf.abs(self.W_spatial), 0) * tf.reduce_sum(tf.abs(self.W_features), 0)
            )
            tf.losses.add_loss(self.readout_sparseness_regularizer, tf.GraphKeys.REGULARIZATION_LOSSES)

            # output nonlinearity
            _, responses, _ = self.data.train()
            b = inv_elu(responses.mean(axis=0))
            self.b_out = tf.get_variable(
                "b_out", shape=[self.data.num_neurons], dtype=tf.float32, initializer=tf.constant_initializer(b)
            )
            self.prediction = tf.identity(
                output_nonlinearity(
                    self.h_out + self.b_out,
                    self.data.num_neurons,
                    vmin=-3,
                    vmax=6,
                    num_bins=50,
                    alpha=output_nonlin_smooth_weight,
                ),
                name="predictions",
            )

            # loss
            self.compute_log_likelihoods(self.prediction, self.responses, self.real_responses)
            tf.losses.add_loss(self.get_log_likelihood())
            total_loss = tf.losses.get_total_loss()
            self.total_loss = total_loss

            # regularizers
            self.smoothness_regularizer = tf.add_n(tf.get_collection("smoothness_regularizer_2d"))
            self.group_sparsity_regularizer = tf.add_n(tf.get_collection("group_sparsity_regularizer_2d"))

            # optimizer
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            # initialize TF session
            self.initialize()

    def get_test_ops(self):
        return [
            self.get_log_likelihood(),
            self.readout_sparseness_regularizer,
            self.group_sparsity_regularizer,
            self.smoothness_regularizer,
            self.total_loss,
            self.prediction,
        ]


class DivisiveNetUnspecificOutputNonlin(Net):
    """Unspecific divisive normalization model"""
    def __init__(
        self,
        filter_sizes,
        out_channels,
        strides,
        paddings,
        smooth_weights,
        sparse_weights,
        readout_sparse_weight,
        output_nonlin_smooth_weight,
        pool_kernel_size,
        pool_type,
        dilation,
        M,
        dn_u_size,
        abs_v_l1_weight,
        dn_padding="SAME",
        data=None,
        log_dir=None,
        log_hash=None,
        global_step=None,
        obs_noise_model="poisson",
        eval_batches=None,
    ):

        super().__init__(
            data=data,
            log_dir=log_dir,
            log_hash=log_hash,
            global_step=global_step,
            obs_noise_model=obs_noise_model,
            eval_batches=eval_batches,
        )
        self.conv = []
        self.W = []
        self.readout_sparseness_regularizer = 0.0

        with self.graph.as_default():
            # convolutional layers
            for i, (filter_size, out_chans, stride, padding, smooth_weight, sparse_weight) in enumerate(
                zip(filter_sizes, out_channels, strides, paddings, smooth_weights, sparse_weights)
            ):
                x = self.images if not i else self.conv[i - 1]
                bn_params = {"decay": 0.9, "is_training": self.is_training, "scale": False}
                scope = "conv{}".format(i)
                reg = lambda w: smoothness_regularizer_2d(w, smooth_weight) + group_sparsity_regularizer_2d(
                    w, sparse_weight
                )
                c = layers.convolution2d(
                    inputs=x,
                    num_outputs=out_chans,
                    kernel_size=int(filter_size),
                    stride=int(stride),
                    padding=padding,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=layers.batch_norm,
                    normalizer_params=bn_params,
                    weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                    weights_regularizer=reg,
                    scope=scope,
                )
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable("weights")
                self.W.append(W)
                self.conv.append(c)

            # Divisive normalization
            self.dn_exponent = tf.abs(
                tf.get_variable(
                    "dn_exponent",
                    shape=[1, 1, 1, self.conv[-1].shape[-1]],
                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.01),
                )
            )
            self.y_n = tf.pow(self.conv[-1], self.dn_exponent)

            if pool_type == "MAX":
                self.pooled = tf.nn.max_pool(self.y_n, pool_kernel_size, strides=[1, 1, 1, 1], padding="SAME")
            elif pool_type == "AVG":
                self.pooled = tf.nn.avg_pool(self.y_n, pool_kernel_size, strides=[1, 1, 1, 1], padding="SAME")
            else:
                raise NotImplementedError

            # get weights for factorized dilated convolution
            # u shape: (space, space, in-features to pool from, out-features being normalized, normalization pools m)
            u_shape = (dn_u_size[0], dn_u_size[1], 1, out_channels[-1], M)
            self.u_unnormalized = tf.Variable(
                initial_value=tf.random_normal(shape=u_shape, stddev=1e-3), trainable=True, name="u"
            )
            self.u = self.u_unnormalized / tf.norm(self.u_unnormalized, ord="fro", axis=[0, 1], keepdims=True)

            # following block is different compared to full model
            self.v = tf.Variable(
                initial_value=tf.random_normal(shape=(1, 1, 1, out_channels[-1], M), stddev=1e-3),
                trainable=True,
                name="v",
            )  # shape: (space, space, in-features to pool from, out-features being normalized, norm. pools M)
            self.v = tf.tile(self.v, [1, 1, out_channels[-1], 1, 1])  # same weights for all in features

            self.abs_u = tf.abs(self.u)
            self.abs_v = tf.abs(self.v)

            # L1 regularizer on abs_v to enforce sparsity (use minimal set of features to normalize)
            self.abs_v_l1_penalty = abs_v_l1_weight * tf.reduce_sum(self.abs_v)
            tf.losses.add_loss(self.abs_v_l1_penalty, tf.GraphKeys.REGULARIZATION_LOSSES)

            self.p_m = self.abs_u * self.abs_v
            self.p = tf.reduce_sum(self.p_m, axis=-1)

            self.supp = tf.nn.conv2d(
                input=self.pooled, filter=self.p, strides=[1, 1, 1, 1], padding=dn_padding, dilations=dilation
            )

            self.semisaturation_const = tf.abs(
                tf.get_variable(
                    "semisaturation_const",
                    shape=[1, 1, 1, self.conv[-1].shape[-1]],
                    initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.01),
                )
            )

            self.abs_eta = tf.abs(self.semisaturation_const)
            self.eta_n = tf.pow(self.abs_eta, self.dn_exponent)

            denom = self.eta_n + self.supp

            # for valid padding in supp convolution: make sure, dimensions for the division fit
            if (dn_padding == "SAME") or (dn_u_size[0] == 1):  # no cropping required
                self.y_n_crop = self.y_n
            else:  # crop to fit self.supp size
                y_n_spatial_crop = (tf.shape(self.y_n)[1] - tf.shape(denom)[1]) // 2
                self.y_n_crop = self.y_n[:, y_n_spatial_crop:-y_n_spatial_crop, y_n_spatial_crop:-y_n_spatial_crop, :]

            denom_greater_zero = tf.Assert(
                tf.reduce_all(tf.greater(denom, 0.0)), [tf.reduce_min(denom), tf.reduce_max(denom)], name="denom"
            )
            with tf.control_dependencies([denom_greater_zero]):
                self.dn_res = self.y_n_crop / denom

            # Readout
            sz = self.dn_res.get_shape()
            px_x_conv = int(sz[1])
            px_y_conv = int(sz[2])
            px_conv = px_x_conv * px_y_conv
            conv_flat = tf.reshape(self.dn_res, [-1, px_conv, out_channels[-1], 1])
            self.W_spatial = tf.abs(
                tf.get_variable(
                    "W_spatial",
                    trainable=True,
                    shape=[px_conv, self.data.num_neurons],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                )
            )
            W_spatial_flat = tf.reshape(self.W_spatial, [px_conv, 1, 1, self.data.num_neurons])
            self.h_spatial = tf.nn.conv2d(
                conv_flat, W_spatial_flat, strides=[1, 1, 1, 1], padding="VALID"
            )  # dot product
            self.W_features = tf.abs(
                tf.get_variable(
                    "W_features",
                    trainable=True,
                    shape=[out_channels[-1], self.data.num_neurons],
                    initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
                )
            )
            self.h_features = self.h_spatial * self.W_features  # element wise product. broadcasting for dimension 0,1
            self.h_out = tf.reduce_sum(self.h_features, [1, 2])

            # L1 regularization for readout layer
            self.readout_sparseness_regularizer = readout_sparse_weight * tf.reduce_sum(
                tf.reduce_sum(tf.abs(self.W_spatial), 0) * tf.reduce_sum(tf.abs(self.W_features), 0)
            )
            tf.losses.add_loss(self.readout_sparseness_regularizer, tf.GraphKeys.REGULARIZATION_LOSSES)

            # output nonlinearity
            _, responses, _ = self.data.train()
            b = inv_elu(responses.mean(axis=0))
            self.b_out = tf.get_variable(
                "b_out", shape=[self.data.num_neurons], dtype=tf.float32, initializer=tf.constant_initializer(b)
            )
            self.prediction = tf.identity(
                output_nonlinearity(
                    self.h_out + self.b_out,
                    self.data.num_neurons,
                    vmin=-3,
                    vmax=6,
                    num_bins=50,
                    alpha=output_nonlin_smooth_weight,
                ),
                name="predictions",
            )

            # loss
            self.compute_log_likelihoods(self.prediction, self.responses, self.real_responses)
            tf.losses.add_loss(self.get_log_likelihood())
            total_loss = tf.losses.get_total_loss()
            self.total_loss = total_loss

            # regularizers
            self.smoothness_regularizer = tf.add_n(tf.get_collection("smoothness_regularizer_2d"))
            self.group_sparsity_regularizer = tf.add_n(tf.get_collection("group_sparsity_regularizer_2d"))

            # optimizer
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

            # initialize TF session
            self.initialize()

    def get_test_ops(self):
        return [
            self.get_log_likelihood(),
            self.readout_sparseness_regularizer,
            self.group_sparsity_regularizer,
            self.smoothness_regularizer,
            self.total_loss,
            self.prediction,
        ]
