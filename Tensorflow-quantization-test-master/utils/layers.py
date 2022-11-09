import tensorflow as tf


def quantize(x):
    abs_value = tf.abs(x)
    vmax = tf.reduce_max(abs_value)
    s = tf.divide(vmax, 127.)
    x = tf.divide(x, s)
    x = tf.math.rint(x)
    # x = tf.math.floor(x)
    return x, s

def quantize_with_scale(x, s):
    #우리가 주는 값은 곱해야됨....
    #x = tf.divide(x, s)
    x = tf.multiply(x, s)
    # x = tf.math.rint(x)
    x = tf.math.floor(x)
    #Clamp
    x = tf.clip_by_value(x, clip_value_min=-127, clip_value_max=127)
    return x

def quantize_with_scale_MAC(x, s):
    #우리가 주는 값은 곱해야됨....
    #x = tf.divide(x, s)
    x = tf.multiply(x, s)
    # x = tf.math.rint(x)
    x = tf.math.floor(x)
    #Clamp
    # x = tf.clip_by_value(x, clip_value_min=-127, clip_value_max=127)
    # x = tf.clip_by_value(x, clip_value_min=-32767, clip_value_max=32767)
    x = tf.clip_by_value(x, clip_value_min=-2147483647, clip_value_max=2147483647)
    return x


def batch_norm(x, mean, variance, offset=None, scale=None):
    return tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=1e-3)


def conv_2d(x, w, b=None, weight_scale=0., strides=1, padding='SAME', dilations=[1,1,1,1], activation=''):
    '''
    2D convolution with quantization (float32-->int8)
    '''
    # quantize input tensor
    x, sx = quantize(x)
    # Actually, convolution compute using float32,
    # because of tensorflow has not supported int8 conv op.
    x = tf.cast(x, dtype=tf.float32)
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
    # multiply scales
    s = sx * weight_scale
    x = x * s
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def conv_2d_bias_too(x, w, b=None, weight_scale=0., strides=1, padding='SAME', dilations=[1,1,1,1], activation=''):
    '''
    2D convolution with quantization (float32-->int8)
    '''
    # quantize input tensor
    x, sx = quantize(x)
    # Actually, convolution compute using float32,
    # because of tensorflow has not supported int8 conv op.
    x = tf.cast(x, dtype=tf.float32)
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
    # multiply scales
    s = sx * weight_scale
    # x = x * s
    if b is not None:
        b=quantize_with_scale_MAC(b,1/s)
        x = tf.nn.bias_add(x, b)
    x = x * s
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x

def conv_2d_with_scale(x, w, b=None, strides=1, padding='SAME', dilations=[1,1,1,1], activation='', input_scale=1.0, weight_scale=1.0):
    '''
    2D convolution with quantization (float32-->int8)
    '''
    # quantize input tensor
    x = quantize_with_scale(x, input_scale)
    # Actually, convolution compute using float32,
    # because of tensorflow has not supported int8 conv op.
    x = tf.cast(x, dtype=tf.float32)
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
    # multiply scales
    
    if b is not None:
        x = tf.nn.bias_add(x, b)
    
    s = input_scale * weight_scale
    #x = x * s
    #우리가 주는 값은 나눠야함...
    x = x / s
    #x = tf.divide(x, s)
    
    if activation == 'relu':
        x = tf.nn.relu(x)
    x = tf.math.rint(x)
    # x = tf.math.floor(x)
    x = tf.clip_by_value(x, clip_value_min=-2147483647, clip_value_max=2147483647)
        
    return x

def conv_2d_with_scale_hardware(x, w, b=None, strides=1, padding='SAME', dilations=[1,1,1,1], activation='', input_scale=1.0, weight_scale=1.0, output_scale=1.0):

    x = tf.cast(x, dtype=tf.float32)
    x = tf.math.floor(x)
    w = tf.math.floor(w)
    
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
    # multiply scales
    
    if b is not None:
        x = tf.nn.bias_add(x, b)
    x = tf.clip_by_value(x, clip_value_min=-2147483647, clip_value_max=2147483647)
    s = output_scale/(input_scale * weight_scale)
    #x = x * s
    #우리가 주는 값은 나눠야함...
    #을 곱셈으로 바꿈
    x_relu = x * s
    # x_relu = tf.math.rint(x_relu)
    x_relu = tf.math.floor(x_relu)
    #x = tf.divide(x, s)
    
    
    x_relu = tf.nn.relu(x_relu)
    
        
    return x, x_relu


def depthwise_conv2d(x, w, b=None, strides=1, padding='SAME', activation=''):
    x = tf.nn.depthwise_conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def separable_conv2d(x, dw, pw, dw_scale=0., pw_scale=0., strides=1, padding='SAME', activation=''):
    x, sx = quantize(x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.nn.separable_conv2d(x, dw, pw, strides=[1, strides, strides, 1], padding=padding)
    # multiply scales
    x = x * sx * dw_scale * pw_scale
    if activation == 'relu':
        x = tf.nn.relu(x)
    return x


def denselayer(x, w, b, weight_scale=0., activation=''):
    x, sx = quantize(x)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.matmul(x, w)
    s = sx * weight_scale
    x = x * s
    x = tf.add(x, b)
    if activation == "relu":
        x = tf.nn.relu(x)
    return x


def denselayer_with_scale(x, w, b, activation='', input_scale=1.0, weight_scale=1.0):
    x = quantize_with_scale(x, input_scale)
    x = tf.cast(x, dtype=tf.float32)
    x = tf.matmul(x, w)
    #s = sx * weight_scale
    #x = x * s
    x = tf.add(x, b)
    s = input_scale * weight_scale
    #x = x * s
    #우리가 주는 값은 나눠야함...
    x = x / s
    #x = tf.divide(x, s)
    if activation == "relu":
        x = tf.nn.relu(x)
    # x = tf.math.rint(x)
    x = tf.math.floor(x)
    x = tf.clip_by_value(x, clip_value_min=-2147483647, clip_value_max=2147483647)
    return x

def denselayer_with_scale_hardware(x, w, b, activation='', input_scale=1.0, weight_scale=1.0, output_scale=1.0):
    x = tf.cast(x, dtype=tf.float32)
    # x = tf.math.rint(x)
    # w = tf.math.rint(w)
    x = tf.math.floor(x)
    w = tf.math.floor(w)
    x = tf.matmul(x, w)
    #s = sx * weight_scale
    #x = x * s
    x = tf.add(x, b)
    x = tf.clip_by_value(x, clip_value_min=-2147483647, clip_value_max=2147483647)
    # x = tf.math.rint(x)
    x = tf.math.floor(x)
    s = output_scale/(input_scale * weight_scale)
    #x = x * s
    #우리가 주는 값은 나눠야함...
    x_relu = x * s
    # x_relu = tf.math.rint(x_relu)
    x_relu = tf.math.floor(x_relu)
    #x = tf.divide(x, s)
    # if activation == "relu":
    x_relu = tf.nn.relu(x_relu)
    return x, x_relu


def maxpool_2d(x, k=2, s=2, padding='VALID'):
    # MaxPool2D wrapper
    x = tf.cast(x, dtype=tf.float32)
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
                          padding=padding)


def avgpool_2d(x, k=2, s=1, padding='VALID'):
    # AvgPool2D wrapper
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1], strides=[1, s, s,1],
                          padding=padding)
