import tensorflow as tf
import keras

from tensorflow.keras.layers import Activation, GlobalMaxPooling2D, Flatten, AveragePooling2D, DepthwiseConv2D, Conv2D, Dense, BatchNormalization, InputLayer, Add, Concatenate, GlobalAveragePooling2D, MaxPooling2D, Multiply, ZeroPadding2D, ReLU
from tensorflow.python.keras.layers.core import TFOpLambda

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions

import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
from tensorflow.python.client import device_lib

print("device 0:" + device_lib.list_local_devices()[0].name)
print("device 1:" + device_lib.list_local_devices()[1].name)
with tf.device('/cpu:0'):
    netname="ResNet50"
    # netname="VGG16"

    imagenet_path='imagenet_valset'
    imagenet_num=50
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
    if netname in ['ResNet50']:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        conv_base = ResNet50(weights='imagenet',classifier_activation=None,include_top=True,input_shape=(224, 224, 3))
    elif netname in ['VGG16']:
        from tensorflow.keras.applications.vgg16 import preprocess_input
        conv_base = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5',include_top=True,input_shape=(224, 224, 3))
        


        


    import cv2
           
    layer_names_qaunt = []
    conv_layers_qaunt = []
    layer_infos_weight_qaunt = []


    f = open("val.txt", 'r')
    x_output = []
    image_num=-1
    top1_v=0
    top5_v=0
    
    max_val = 0
    image_num=0
    for item in imagenet_filelist:
        image_num=image_num+1
        if image_num<=1000:
            # x_input = load_img(imagenet_path+'/'+item, target_size=(224,224)) #0.0~1.0, BGR
            # x_input = img_to_array(x_input)
            
            x_input = cv2.imread(imagenet_path+'/'+item)
            x_input = cv2.resize(x_input, (256, 256))
            # x_input = cv2.cvtColor(x_input, cv2.COLOR_BGR2RGB)
            
            mid_x, mid_y = 256//2, 256//2
            offset_x, offset_y = 224//2, 224//2
            
            x_input = x_input[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
        
            x_input = x_input[...,::-1].astype(np.float32)
            x_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1], x_input.shape[2])
                
            x_input = preprocess_input(x_input) #make 0.0~1.0 to -1.0~1.0?
            np.savetxt(os.path.join("dump2.txt"), np.array(x_input).transpose(0,3,1,2).flatten())
            x_output_temp= conv_base.predict(x_input)
            
            top1=np.argmax(x_output_temp)
            x_output_temp[0][top1]=-2
            top2=np.argmax(x_output_temp)
            x_output_temp[0][top2]=-2
            top3=np.argmax(x_output_temp)
            x_output_temp[0][top3]=-2
            top4=np.argmax(x_output_temp)
            x_output_temp[0][top4]=-2
            top5=np.argmax(x_output_temp)
            
            
            
            line = f.readline()
            line = line.strip()
            line = line.split(' ')[1]
            
            label = int(line)
            print(label)
            print(top1)
            print(top2)
            print(top3)
            print(top4)
            print(top5)
            if top1==label:
                top1_v+=1
                top5_v+=1
            if top2==label:
                top5_v+=1
            if top3==label:
                top5_v+=1
            if top4==label:
                top5_v+=1
            if top5==label:
                top5_v+=1
            print(top1_v/(image_num))
            print(top5_v/(image_num))
            
        else:
            break
       
