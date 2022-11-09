from utils.layers import *
import tensorflow as tf
import numpy as np


def quantize(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s

def quantize_tf(x):
    abs_value = tf.abs(x)
    vmax = tf.reduce_max(abs_value)
    s = tf.divide(vmax, 127.)
    x = tf.divide(x, s)
    x = tf.math.rint(x)
    return x, s

def get_weights_biases(weights, weight_name, bias_name='bbb', quant=True):
    w = tf.constant(weights[weight_name], dtype=tf.float32)
    try:
        b = tf.constant(weights[bias_name], dtype=tf.float32)
    except:
        b = None
    return w, b

def get_weights_biases_scale(weights, weight_name, bias_name='bbb', quant=True):
    w = weights[weight_name]
    if quant:
        w, s = quantize(w)
        w = tf.constant(w, dtype=tf.float32)
    else:
        w = tf.constant(weights[weight_name], dtype=tf.float32)
        s = 0.
    try:
        b = tf.constant(weights[bias_name], dtype=tf.float32)
    except:
        b = None
    return w, b, s

def get_bn_param(weights, mean, std, beta, gamma):
    mean = tf.constant(weights[mean], dtype=tf.float32)
    std = tf.constant(weights[std], dtype=tf.float32)
    beta = tf.constant(weights[beta], dtype=tf.float32)
    gamma = tf.constant(weights[gamma], dtype=tf.float32)
    return mean, std, beta, gamma

def calc_weights_biases_scale(w, quant=True):
    if quant:
        w, s = quantize_tf(w)
        # w = tf.constant(w, dtype=tf.float32)
    else:
        # w = tf.constant(w, dtype=tf.float32)
        s = 0.
    return w, s

def bn_folding (w, b, mean, std, beta, gamma):
    eps = 1.001e-5 #ResNet50
    # eps = 1e-3 #MobileNet
    
    #fused_weight = np.multiply(w,gamma) / np.sqrt(std + eps)
    #fused_bias = (np.multiply(np.subtract(b,mean) ,gamma) / np.sqrt(std +eps)) + beta
    f_w = tf.math.multiply(w,gamma) / tf.math.sqrt(std + eps)
    f_b = (tf.math.multiply(tf.math.subtract(b,mean),gamma) / tf.math.sqrt(std + eps)) + beta
    return f_w, f_b

def get_weight (weights, conv, bn):
    bn_params = ['_running_mean:0', '_running_std:0', '_beta:0', '_gamma:0']
    conv_wb = ['_W:0', '_b:0']
    
    w, b = get_weights_biases(weights, conv + conv_wb[0], conv + conv_wb[1])
    mean, std, beta, gamma = get_bn_param(weights, bn + bn_params[0], bn + bn_params[1], bn + bn_params[2], bn + bn_params[3])
    w, b = bn_folding(w, b, mean, std, beta, gamma) 
    
    return w, b

def identity_block(inputs, weights, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_names = ['2a', '2b', '2c']

    conv = conv_name_base + conv_names[0]
    bn = bn_name_base + conv_names[0]
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    
    x = conv_2d(inputs, w, b, s)
    x = tf.nn.relu(x)

    conv = conv_name_base + conv_names[1]
    bn = bn_name_base + conv_names[1]
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    
    x = conv_2d(x, w, b, s)
    x = tf.nn.relu(x)
    
    conv = conv_name_base + conv_names[2]
    bn = bn_name_base + conv_names[2]
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    
    x = conv_2d(x, w, b, s)

    x = tf.add(x, inputs)
    return tf.nn.relu(x)


def conv_block(inputs, weights, stage, block, strides=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    conv_names = ['2a', '2b', '2c']

    conv = conv_name_base + conv_names[0]
    bn = bn_name_base + conv_names[0]
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    
    x = conv_2d(inputs, w, b, s, strides=strides)
    x = tf.nn.relu(x)

    conv = conv_name_base + conv_names[1]
    bn = bn_name_base + conv_names[1]
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    
    x = conv_2d(x, w, b, s)
    x = tf.nn.relu(x)

    conv = conv_name_base + conv_names[2]
    bn = bn_name_base + conv_names[2]
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    
    x = conv_2d(x, w, b, s)

    # shortcut
    conv = conv_name_base + '1'
    bn = bn_name_base + '1'
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    
    shortcut = conv_2d(inputs, w, b, s, strides=strides)
    
    x = tf.add(x, shortcut)
    return tf.nn.relu(x)


def ResNet50(x, weights):
    # init convolution
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
    conv = 'conv1'
    bn = 'bn_conv1'
    w, b = get_weight(weights, conv, bn)
    w, s = calc_weights_biases_scale(w)
    x = conv_2d(x, w, b, s, strides=2)

    x = tf.nn.relu(x)
    x = maxpool_2d(x, k=3, s=2, padding='SAME')

    x = conv_block(x, weights, stage=2, block='a', strides=1)
    x = identity_block(x, weights, stage=2, block='b')
    x = identity_block(x, weights, stage=2, block='c')

    x = conv_block(x, weights, stage=3, block='a')
    x = identity_block(x, weights, stage=3, block='b')
    x = identity_block(x, weights, stage=3, block='c')
    x = identity_block(x, weights, stage=3, block='d')

    x = conv_block(x, weights, stage=4, block='a')
    x = identity_block(x, weights, stage=4, block='b')
    x = identity_block(x, weights, stage=4, block='c')
    x = identity_block(x, weights, stage=4, block='d')
    x = identity_block(x, weights, stage=4, block='e')
    x = identity_block(x, weights, stage=4, block='f')

    x = conv_block(x, weights, stage=5, block='a')
    x = identity_block(x, weights, stage=5, block='b')
    x = identity_block(x, weights, stage=5, block='c')

    x = avgpool_2d(x, k=7)

    w, b, s = get_weights_biases_scale(weights, 'fc1000_W:0', 'fc1000_b:0')
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = denselayer(x, w, b, s)
    return x
