import tensorflow as tf
import keras

#from tensorflow.keras.layers import Activation, GlobalMaxPooling2D, Flatten, AveragePooling2D, DepthwiseConv2D, Conv2D, Dense, BatchNormalization, InputLayer, Add, Concatenate, GlobalAveragePooling2D, MaxPooling2D, Multiply, ZeroPadding2D, ReLU
#from tensorflow.python.keras.layers.core import TFOpLambda

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

#from tensorflow.keras.applications.vgg16 import VGG16
#from tensorflow.keras.applications.vgg16 import decode_predictions

#from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.resnet50 import decode_predictions

#import argparse
from models import resnet50, vgg16,vgg16_q, inception_v3, mobilenet, xception, squeezenet
from utils.load_weights import weight_loader
from pkl_reader import DataGenerator
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
from tensorflow.python.client import device_lib

def top5_acc(pred, k=5):
    Inf = 0.
    results = []
    for i in range(k):
        results.append(pred.index(max(pred)))
        pred[pred.index(max(pred))] = Inf
    return results


print("device 0:" + device_lib.list_local_devices()[0].name)
print("device 1:" + device_lib.list_local_devices()[1].name)
with tf.device('/gpu:0'):
    # netname="ResNet50"
    netname="VGG16"

    imagenet_path='../imagenet_valset'
    imagenet_num=5000
    imagenet_filelist = os.listdir(imagenet_path)
    imagenet_filelist.sort()

    pathname=netname+"/"
    '''
    def preprocess_input(im):
        # im = im[..., ::-1]
        im[..., 0] -= 103.939
        im[..., 1] -= 116.779
        im[..., 2] -= 123.68
        return im
    '''
    weights = weight_loader('./weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5'.format(weights[model_name]))
    if netname in ['ResNet50']:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        # conv_base = ResNet50('vgg16_weights_tf_dim_ordering_tf_kernels.h5',include_top=True,input_shape=(224, 224, 3))
    elif netname in ['VGG16']:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        X = tf.placeholder(tf.float32, [None, 224, 224, 3])
        conv_base = vgg16_q.VGG16(X, weights)
        #conv_base = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels',include_top=True,input_shape=(224, 224, 3))
        
    #conv_base.summary()
    #데이터:CHW(RS)
    #바이어스:K
    #내부순서가중치:KCRS

    #layer_config = []
    #layer_config = conv_base.get_config()
    #print(layer_config)


        

    import cv2
           
    layer_names_qaunt = []
    conv_layers_qaunt = []
    layer_infos_weight_qaunt = []


    layer_infos_active_quant = []
    quantize_level = 4096
    print("====================================")
    print("Activation quantization")
    image_num=0

    f = open("val.txt", 'r')
    
    top1=0
    top5=0

    for item in imagenet_filelist:
        image_num=image_num+1
        if image_num<=imagenet_num:
            x_input = load_img(imagenet_path+'/'+item, target_size=(224,224)) #0.0~1.0, BGR
            x_input = img_to_array(x_input)
            
            # x_input = cv2.imread(imagenet_path+'/'+item)
            # x_input = cv2.resize(x_input,(224,224))
            # x_input = cv2.cvtColor(x_input, cv2.COLOR_BGR2RGB)
            # x_input = np.asarray(x_input)
                        
    #           test=array_to_img(x_input)
            # x_input=x_input/1.0
    #            cv2.imshow("test",test)
            
            x_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1], x_input.shape[2])
            
            x_input = preprocess_input(x_input) #make 0.0~1.0 to -1.0~1.0?
            x_output= conv_base.predict(x_input)
            
            line = f.readline()
            line = line.strip()
            line = line.split(' ')[1]
            print(line)
            
            print(np.argmax(x_output))
            if(int(np.argmax(x_output))==int(line)):
                top1+=1
                top5+=1
            #x_output=np.where(x_output==np.max(x_output),-1,x_output)
            x_output[0][np.argmax(x_output)]=-2
            print(np.argmax(x_output))
            if(int(np.argmax(x_output))==int(line)):
                top5+=1
            #x_output=np.where(x_output==np.max(x_output),-1,x_output)
            x_output[0][np.argmax(x_output)]=-2
            print(np.argmax(x_output))
            if(int(np.argmax(x_output))==int(line)):
                top5+=1
            #x_output=np.where(x_output==np.max(x_output),-1,x_output)
            x_output[0][np.argmax(x_output)]=-2
            print(np.argmax(x_output))
            if(int(np.argmax(x_output))==int(line)):
                top5+=1
            #x_output=np.where(x_output==np.max(x_output),-1,x_output)
            x_output[0][np.argmax(x_output)]=-2
            print(np.argmax(x_output))
            if(int(np.argmax(x_output))==int(line)):
                top5+=1
            print(top1/image_num)
            print(top5/image_num)
            
            
            print("===")
       
