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

def quantize_with_scale(weights, scale):
    #우리가 주는 값은 곱해야함...
    # qweights = weights / s
    qweights = weights * scale
    # qweights = np.round(qweights)
    qweights = np.round(qweights)
    #CLAMP
    qweights = np.where(qweights > 127, 127, qweights)
    qweights = np.where(qweights < -127, -127, qweights)
    qweights = qweights.astype(np.int8)
    return qweights

def quantize_with_scale_t(x, s):
    #우리가 주는 값은 곱해야됨....
    #x = tf.divide(x, s)
    x = tf.multiply(x, s)
    # x = tf.math.rint(x)
    x_upper = tf.math.floor(x)
    x_upper = tf.clip_by_value(x_upper, clip_value_min=0, clip_value_max=127)
    x_bottom = tf.math.ceil(x)
    x_bottom = tf.clip_by_value(x_bottom, clip_value_min=-127, clip_value_max=0)
    x=x_upper+x_bottom
    #Clamp
    return x

def quantize_with_scale_without_clamp(weights, scale):
    #우리가 주는 값은 곱해야함...
    # qweights = weights / s
    qweights = weights * scale
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int32)
    return qweights


def get_weights_biases(weights, weight_name, bias_name='bbb', quant=True):
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

def get_weights_biases_with_scale(weights, weight_name, bias_name='bbb', input_scale=1.0, weight_scale=1.0):
    w = weights[weight_name]
    w = quantize_with_scale(w, weight_scale)
    w = tf.constant(w, dtype=tf.float32)
    
    try:
        # b = tf.constant(weights[bias_name], dtype=tf.float32)
        b = weights[bias_name]
        b = quantize_with_scale(b, input_scale* weight_scale)
        b = tf.constant(b, dtype=tf.float32)   
    except:
        b = None
    return w, b

'''
def VGG16(x, weights):
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
    # block 1
    w, b = get_weights_biases_with_scale(weights, 'block1_conv1_W_1:0', 'block1_conv1_b_1:0', input_scale=0.8407199482339983, weight_scale=189.15696562852443)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.8407199482339983, weight_scale=189.15696562852443)
    # w, b, s = get_weights_biases(weights, 'block1_conv1_W_1:0', 'block1_conv1_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    w, b = get_weights_biases_with_scale(weights, 'block1_conv2_W_1:0', 'block1_conv2_b_1:0', input_scale=0.17318415597238515, weight_scale=439.18657705421447)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.17318415597238515, weight_scale=439.18657705421447)
    # w, b, s = get_weights_biases(weights, 'block1_conv2_W_1:0', 'block1_conv2_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # block2
    w, b = get_weights_biases_with_scale(weights, 'block2_conv1_W_1:0', 'block2_conv1_b_1:0', input_scale=0.03637984382118375, weight_scale=304.8406715487954)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.03637984382118375, weight_scale=304.8406715487954)
    # w, b, s = get_weights_biases(weights, 'block2_conv1_W_1:0', 'block2_conv1_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    w, b = get_weights_biases_with_scale(weights, 'block2_conv2_W_1:0', 'block2_conv2_b_1:0', input_scale=0.02069920158798699, weight_scale=457.8627011282144)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.02069920158798699, weight_scale=457.8627011282144)
    # w, b, s = get_weights_biases(weights, 'block2_conv2_W_1:0', 'block2_conv2_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # block3
    w, b = get_weights_biases_with_scale(weights, 'block3_conv1_W_1:0', 'block3_conv1_b_1:0', input_scale=0.014498163197449872, weight_scale=233.27971124072639)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.014498163197449872, weight_scale=233.27971124072639)
    # w, b, s = get_weights_biases(weights, 'block3_conv1_W_1:0', 'block3_conv1_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    w, b = get_weights_biases_with_scale(weights, 'block3_conv2_W_1:0', 'block3_conv2_b_1:0', input_scale=0.01031102837042495, weight_scale=276.49789065071536)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.01031102837042495, weight_scale=276.49789065071536)
    # w, b, s = get_weights_biases(weights, 'block3_conv2_W_1:0', 'block3_conv2_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    w, b = get_weights_biases_with_scale(weights, 'block3_conv3_W_1:0', 'block3_conv3_b_1:0', input_scale=0.01179473180769183, weight_scale=324.36239879785177)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.01179473180769183, weight_scale=324.36239879785177)
    # w, b, s = get_weights_biases(weights, 'block3_conv3_W_1:0', 'block3_conv3_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    # block4
    # w, b, s = get_weights_biases(weights, 'block4_conv1_W_1:0', 'block4_conv1_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    # w, b, s = get_weights_biases(weights, 'block4_conv2_W_1:0', 'block4_conv2_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    # w, b, s = get_weights_biases(weights, 'block4_conv3_W_1:0', 'block4_conv3_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')
    
    w, b = get_weights_biases_with_scale(weights, 'block4_conv1_W_1:0', 'block4_conv1_b_1:0', input_scale=0.01226605669640658, weight_scale=404.6103127559724)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.01226605669640658, weight_scale=404.6103127559724)

    w, b = get_weights_biases_with_scale(weights, 'block4_conv2_W_1:0', 'block4_conv2_b_1:0', input_scale=0.01869454741771969, weight_scale=376.1179722135677)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.01869454741771969, weight_scale=376.1179722135677)

    w, b = get_weights_biases_with_scale(weights, 'block4_conv3_W_1:0', 'block4_conv3_b_1:0', input_scale=0.03333508444279601, weight_scale=495.63737885916515)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.03333508444279601, weight_scale=495.63737885916515)
    
    x = maxpool_2d(x, k=2, s=2)

    # block5
    w, b = get_weights_biases_with_scale(weights, 'block5_conv1_W_1:0', 'block5_conv1_b_1:0', input_scale=0.04545985810657578, weight_scale=666.8613168476023)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.04545985810657578, weight_scale=666.8613168476023)

    w, b = get_weights_biases_with_scale(weights, 'block5_conv2_W_1:0', 'block5_conv2_b_1:0', input_scale=0.10852119047953133, weight_scale=619.0553781526582)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.10852119047953133, weight_scale=619.0553781526582)

    w, b = get_weights_biases_with_scale(weights, 'block5_conv3_W_1:0', 'block5_conv3_b_1:0', input_scale=0.2068618862301402, weight_scale=442.5138565795882)
    x = conv_2d_with_scale(x, w, b, activation='relu', input_scale=0.2068618862301402, weight_scale=442.5138565795882)

    
    # w, b, s = get_weights_biases(weights, 'block5_conv1_W_1:0', 'block5_conv1_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    # w, b, s = get_weights_biases(weights, 'block5_conv2_W_1:0', 'block5_conv2_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')

    # w, b, s = get_weights_biases(weights, 'block5_conv3_W_1:0', 'block5_conv3_b_1:0')
    # x = conv_2d(x, w, b, s, activation='relu')
    x = maxpool_2d(x, k=2, s=2)

    
    # fc1
    # w, b, s = get_weights_biases(weights, 'fc1_W_1:0', 'fc1_b_1:0')
    # x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])

    # x = tf.matmul(x, w)
    # x = tf.add(x, b)
    # x = tf.nn.relu(x)

    # x = denselayer(x, w, b, s, activation='relu')
    
    w, b = get_weights_biases_with_scale(weights, 'fc1_W_1:0', 'fc1_b_1:0', input_scale=0.4016046384204797, weight_scale=4692.909141028782)
    x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
    x = denselayer_with_scale(x, w, b, activation='relu', input_scale=0.4016046384204797, weight_scale=4692.909141028782)

    # fc2
    # w, b, s = get_weights_biases(weights, 'fc2_W_1:0', 'fc2_b_1:0')
    # x = tf.matmul(x, w)
    # x = tf.add(x, b)
    # x = tf.nn.relu(x)
    
    # x = denselayer(x, w, b, s, activation='relu')
   
    w, b = get_weights_biases_with_scale(weights, 'fc2_W_1:0', 'fc2_b_1:0', input_scale=3.4070641141978646, weight_scale=3893.2225357008797)
    x = denselayer_with_scale(x, w, b, activation='relu', input_scale=3.4070641141978646, weight_scale=3893.2225357008797)

    # fc3
    # w, b, s = get_weights_biases(weights, 'predictions_W_1:0', 'predictions_b_1:0')
    
    # x = tf.matmul(x, w)
    # x = tf.add(x, b)
    
    # x = denselayer(x, w, b, s)
    
    w, b = get_weights_biases_with_scale(weights, 'predictions_W_1:0', 'predictions_b_1:0', input_scale=10.72910424142159, weight_scale=2218.1222124965752)
    x = denselayer_with_scale(x, w, b, input_scale=10.72910424142159, weight_scale=2218.1222124965752)
    return x


'''
def VGG16(x, weights):
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
    x_relu = quantize_with_scale_t(x,0.8407199482339983)

    
    w, b = get_weights_biases_with_scale(weights, 'block1_conv1_W_1:0', 'block1_conv1_b_1:0', input_scale=0.8407199482339983, weight_scale=189.15696562852443)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.8407199482339983, weight_scale=189.15696562852443, output_scale=0.13249275737007907)


    w, b = get_weights_biases_with_scale(weights, 'block1_conv2_W_1:0', 'block1_conv2_b_1:0', input_scale=0.13249275737007907, weight_scale=439.18657705421447)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.13249275737007907, weight_scale=439.18657705421447, output_scale=0.025170243253263266)


    x_relu = maxpool_2d(x_relu, k=2, s=2)


    # block2
    w, b = get_weights_biases_with_scale(weights, 'block2_conv1_W_1:0', 'block2_conv1_b_1:0', input_scale=0.025170243253263266, weight_scale=304.8406715487954)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.025170243253263266, weight_scale=304.8406715487954, output_scale=0.015469715678311863)


    w, b = get_weights_biases_with_scale(weights, 'block2_conv2_W_1:0', 'block2_conv2_b_1:0', input_scale=0.015469715678311863, weight_scale=457.8627011282144)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.015469715678311863, weight_scale=457.8627011282144, output_scale=0.008480955133647251)

    
    x_relu = maxpool_2d(x_relu, k=2, s=2)


    # block3
    w, b = get_weights_biases_with_scale(weights, 'block3_conv1_W_1:0', 'block3_conv1_b_1:0', input_scale=0.008480955133647251, weight_scale=233.27971124072639)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.008480955133647251, weight_scale=233.27971124072639, output_scale=0.006243142682327999)


    w, b = get_weights_biases_with_scale(weights, 'block3_conv2_W_1:0', 'block3_conv2_b_1:0', input_scale=0.006243142682327999, weight_scale=276.49789065071536)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.006243142682327999, weight_scale=276.49789065071536, output_scale=0.00769625222583729)


    w, b = get_weights_biases_with_scale(weights, 'block3_conv3_W_1:0', 'block3_conv3_b_1:0', input_scale=0.00769625222583729, weight_scale=324.36239879785177)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.00769625222583729, weight_scale=324.36239879785177, output_scale=0.007259157263787016)

    
    x_relu = maxpool_2d(x_relu, k=2, s=2)


    # block4    
    w, b = get_weights_biases_with_scale(weights, 'block4_conv1_W_1:0', 'block4_conv1_b_1:0', input_scale=0.007259157263787016, weight_scale=404.6103127559724)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.007259157263787016, weight_scale=404.6103127559724, output_scale=0.009999476811150517)


    w, b = get_weights_biases_with_scale(weights, 'block4_conv2_W_1:0', 'block4_conv2_b_1:0', input_scale=0.009999476811150517, weight_scale=376.1179722135677)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.009999476811150517, weight_scale=376.1179722135677, output_scale=0.017616681840153945)


    w, b = get_weights_biases_with_scale(weights, 'block4_conv3_W_1:0', 'block4_conv3_b_1:0', input_scale=0.017616681840153945, weight_scale=495.63737885916515)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.017616681840153945, weight_scale=495.63737885916515, output_scale=0.03088123962834647)

    
    x_relu = maxpool_2d(x_relu, k=2, s=2)


    # block5
    w, b = get_weights_biases_with_scale(weights, 'block5_conv1_W_1:0', 'block5_conv1_b_1:0', input_scale=0.03088123962834647, weight_scale=666.8613168476023)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.03088123962834647, weight_scale=666.8613168476023, output_scale=0.05189479932678962)


    w, b = get_weights_biases_with_scale(weights, 'block5_conv2_W_1:0', 'block5_conv2_b_1:0', input_scale=0.05189479932678962, weight_scale=619.0553781526582)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.05189479932678962, weight_scale=619.0553781526582, output_scale=0.11665463969797243)


    w, b = get_weights_biases_with_scale(weights, 'block5_conv3_W_1:0', 'block5_conv3_b_1:0', input_scale=0.11665463969797243, weight_scale=442.5138565795882)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.11665463969797243, weight_scale=442.5138565795882, output_scale=0.19932758112915577)


    x_relu = maxpool_2d(x_relu, k=2, s=2)

    
    # fc1
    w, b = get_weights_biases_with_scale(weights, 'fc1_W_1:0', 'fc1_b_1:0', input_scale=0.19932758112915577, weight_scale=4692.909141028782)
    x_relu = tf.reshape(x_relu, [-1, w.get_shape().as_list()[0]])
    x, x_relu = denselayer_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.19932758112915577, weight_scale=4692.909141028782, output_scale=1.6241749102288492)


    # fc2
    w, b = get_weights_biases_with_scale(weights, 'fc2_W_1:0', 'fc2_b_1:0', input_scale=1.6241749102288492, weight_scale=3893.2225357008797)
    x, x_relu = denselayer_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=1.6241749102288492, weight_scale=3893.2225357008797, output_scale=6.171023945140827)


    # fc3
    
    w, b = get_weights_biases_with_scale(weights, 'predictions_W_1:0', 'predictions_b_1:0', input_scale=6.171023945140827, weight_scale=2218.1222124965752)
    x, x_relu = denselayer_with_scale_hardware(x_relu, w, b, input_scale=6.171023945140827, weight_scale=2218.1222124965752, output_scale=127.0)
    x_relu=x_relu/127.0
    
    
    return x_relu



def VGG16_a(x, weights):
    x = tf.reshape(x, shape=[-1, 224, 224, 3])
    x_relu = quantize_with_scale_t(x,0.8407199482339983)
    # x_relu=x_relu/0.8407199482339983
    
    w, b = get_weights_biases_with_scale(weights, 'block1_conv1_W_1:0', 'block1_conv1_b_1:0', input_scale=0.8407199482339983, weight_scale=189.15696562852443)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.8407199482339983, weight_scale=189.15696562852443, output_scale=0.13249275737007907)
    # x_relu=x_relu/0.13249275737007907

    w, b = get_weights_biases_with_scale(weights, 'block1_conv2_W_1:0', 'block1_conv2_b_1:0', input_scale=0.13249275737007907, weight_scale=439.18657705421447)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.13249275737007907, weight_scale=439.18657705421447, output_scale=0.025170243253263266)
    # x_relu=x_relu/0.025170243253263266

    x_relu = maxpool_2d(x_relu, k=2, s=2)
    # x_relu=x_relu/0.025170243253263266

    # block2
    w, b = get_weights_biases_with_scale(weights, 'block2_conv1_W_1:0', 'block2_conv1_b_1:0', input_scale=0.025170243253263266, weight_scale=304.8406715487954)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.025170243253263266, weight_scale=304.8406715487954, output_scale=0.015469715678311863)
    # x_relu=x_relu/0.015469715678311863

    w, b = get_weights_biases_with_scale(weights, 'block2_conv2_W_1:0', 'block2_conv2_b_1:0', input_scale=0.015469715678311863, weight_scale=457.8627011282144)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.015469715678311863, weight_scale=457.8627011282144, output_scale=0.008480955133647251)
    # x_relu=x_relu/0.008480955133647251
    
    x_relu = maxpool_2d(x_relu, k=2, s=2)
    # x_relu=x_relu/0.008480955133647251

    # block3
    w, b = get_weights_biases_with_scale(weights, 'block3_conv1_W_1:0', 'block3_conv1_b_1:0', input_scale=0.008480955133647251, weight_scale=233.27971124072639)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.008480955133647251, weight_scale=233.27971124072639, output_scale=0.006243142682327999)
    # x_relu=x_relu/0.006243142682327999

    w, b = get_weights_biases_with_scale(weights, 'block3_conv2_W_1:0', 'block3_conv2_b_1:0', input_scale=0.006243142682327999, weight_scale=276.49789065071536)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.006243142682327999, weight_scale=276.49789065071536, output_scale=0.00769625222583729)
    # x_relu=x_relu/0.00769625222583729

    w, b = get_weights_biases_with_scale(weights, 'block3_conv3_W_1:0', 'block3_conv3_b_1:0', input_scale=0.00769625222583729, weight_scale=324.36239879785177)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.00769625222583729, weight_scale=324.36239879785177, output_scale=0.007259157263787016)
    #x_relu=x_relu/0.007259157263787016
    #위번 레8잉 ㅓ 
    
    
    x_relu = maxpool_2d(x_relu, k=2, s=2)
    # x_relu=x_relu/0.007259157263787016

    # block4    
    w, b = get_weights_biases_with_scale(weights, 'block4_conv1_W_1:0', 'block4_conv1_b_1:0', input_scale=0.007259157263787016, weight_scale=404.6103127559724)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.007259157263787016, weight_scale=404.6103127559724, output_scale=0.009999476811150517)
    # x_relu=x_relu/0.009999476811150517

    w, b = get_weights_biases_with_scale(weights, 'block4_conv2_W_1:0', 'block4_conv2_b_1:0', input_scale=0.009999476811150517, weight_scale=376.1179722135677)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.009999476811150517, weight_scale=376.1179722135677, output_scale=0.017616681840153945)
    # x_relu=x_relu/0.017616681840153945

    w, b = get_weights_biases_with_scale(weights, 'block4_conv3_W_1:0', 'block4_conv3_b_1:0', input_scale=0.017616681840153945, weight_scale=495.63737885916515)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.017616681840153945, weight_scale=495.63737885916515, output_scale=0.03088123962834647)
    # x_relu=x_relu/0.03088123962834647
    #12버ㅏㄴ 레이어 
    x_relu = maxpool_2d(x_relu, k=2, s=2)
    # x_relu=x_relu/0.03088123962834647

    # block5
    w, b = get_weights_biases_with_scale(weights, 'block5_conv1_W_1:0', 'block5_conv1_b_1:0', input_scale=0.03088123962834647, weight_scale=666.8613168476023)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.03088123962834647, weight_scale=666.8613168476023, output_scale=0.05189479932678962)
    # x_relu=x_relu/0.05189479932678962

    w, b = get_weights_biases_with_scale(weights, 'block5_conv2_W_1:0', 'block5_conv2_b_1:0', input_scale=0.05189479932678962, weight_scale=619.0553781526582)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.05189479932678962, weight_scale=619.0553781526582, output_scale=0.11665463969797243)
    # x_relu=x_relu/0.11665463969797243

    w, b = get_weights_biases_with_scale(weights, 'block5_conv3_W_1:0', 'block5_conv3_b_1:0', input_scale=0.11665463969797243, weight_scale=442.5138565795882)
    x, x_relu = conv_2d_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.11665463969797243, weight_scale=442.5138565795882, output_scale=0.19932758112915577)
    # x_relu=x_relu/0.19932758112915577

    x_relu = maxpool_2d(x_relu, k=2, s=2)
    # x_relu=x_relu/0.25375007618526124
    
    # fc1
    w, b = get_weights_biases_with_scale(weights, 'fc1_W_1:0', 'fc1_b_1:0', input_scale=0.19932758112915577, weight_scale=4692.909141028782)
    x_relu = tf.reshape(x_relu, [-1, w.get_shape().as_list()[0]])
    x, x_relu = denselayer_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=0.19932758112915577, weight_scale=4692.909141028782, output_scale=1.6241749102288492)
    # x_relu=x_relu/1.6241749102288492

    # fc2
    w, b = get_weights_biases_with_scale(weights, 'fc2_W_1:0', 'fc2_b_1:0', input_scale=1.6241749102288492, weight_scale=3893.2225357008797)
    x, x_relu = denselayer_with_scale_hardware(x_relu, w, b, activation='relu', input_scale=1.6241749102288492, weight_scale=3893.2225357008797, output_scale=6.171023945140827)
    # x_relu=x_relu/6.171023945140827

    # fc3
    
    w, b = get_weights_biases_with_scale(weights, 'predictions_W_1:0', 'predictions_b_1:0', input_scale=6.171023945140827, weight_scale=2218.1222124965752)
    x, x_relu = denselayer_with_scale_hardware(x_relu, w, b, input_scale=6.171023945140827, weight_scale=2218.1222124965752, output_scale=127.0)
    x_relu=x_relu/127.0
    
    
    
    return x_relu
    # return x
