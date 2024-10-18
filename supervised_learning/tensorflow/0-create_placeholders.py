#!/usr/bin/env python3
"""task1"""
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """function that returns two placeholders"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
