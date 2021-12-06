import tensorflow as tf


def flops_loss(flops, max_flops):
    return abs(max_flops - flops)


def binary_crossentr_flops(flops, max_flops):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce + flops_loss(flops, max_flops)


def categorical_crossentr_flops(flops, max_flops):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return cce + flops_loss(flops, max_flops)


def sparse_categorical_crossentr_flops(flops, max_flops):
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    return cce + flops_loss(flops, max_flops)