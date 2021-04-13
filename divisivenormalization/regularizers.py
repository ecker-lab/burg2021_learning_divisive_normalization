import tensorflow as tf


def smoothness_regularizer_2d(W, weight=1.0):
    with tf.variable_scope("smoothness"):
        lap = tf.constant([[0.25, 0.5, 0.25], [0.5, -3.0, 0.5], [0.25, 0.5, 0.25]])
        lap = tf.expand_dims(tf.expand_dims(lap, 2), 3)
        out_channels = W.get_shape().as_list()[2]
        W_lap = tf.nn.depthwise_conv2d(
            tf.transpose(W, perm=[3, 0, 1, 2]),
            tf.tile(lap, [1, 1, out_channels, 1]),
            strides=[1, 1, 1, 1],
            padding="SAME",
        )
        penalty = tf.reduce_sum(
            tf.reduce_sum(tf.square(W_lap), [1, 2]) / tf.transpose(tf.reduce_sum(tf.square(W), [0, 1]))
        )
        penalty = tf.identity(weight * penalty, name="penalty")
        tf.add_to_collection("smoothness_regularizer_2d", penalty)
        return penalty


def group_sparsity_regularizer_2d(W, weight=1.0):
    with tf.variable_scope("group_sparsity"):
        penalty = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(W), [0, 1])))
        penalty = tf.identity(weight * penalty, name="penalty")
        tf.add_to_collection("group_sparsity_regularizer_2d", penalty)
        return penalty


def smoothness_regularizer_1d(w, weight=1.0, order=2):
    penalty = 0
    kernel = tf.constant([-1.0, 1.0], shape=[2, 1, 1], dtype=tf.float32)
    for _ in range(order):
        w = tf.nn.conv1d(w, kernel, 1, "VALID")
        penalty += tf.reduce_sum(tf.reduce_mean(tf.square(w), 1))
    penalty = tf.identity(weight * penalty, name="penalty")
    tf.add_to_collection("smoothness_regularizer_1d", penalty)
    return penalty
