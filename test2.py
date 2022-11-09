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
        
    #conv_base.summary()
    #데이터:CHW(RS)
    #바이어스:K
    #내부순서가중치:KCRS

    #layer_config = []
    #layer_config = conv_base.get_config()
    #print(layer_config)

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)

        
    conv_layers = []
    layer_names = []

    import cv2
           
    layer_names_qaunt = []
    conv_layers_qaunt = []
    layer_infos_weight_qaunt = []


    layer_infos_active_quant = []
    quantize_level = 4096
    print("====================================")
    print("Activation quantization")
    intermediate_layer_model = []
    layer_num=-1

    for l in conv_base.layers:
        print("====================================")
        layer_num=layer_num+1
        skip_bool=False
        print("layer name : " + str(l.name))
        if netname in ['VGG16']:
            if isinstance(l, Flatten):
                skip_bool=True        
        #if netname in ['ResNet50']:
            #if isinstance(l, GlobalAveragePooling2D):
            #    skip_bool=True  
        if skip_bool is True:
            print("skip")
            continue

        intermediate_layer_model=tf.keras.Model(inputs=conv_base.input, outputs=conv_base.layers[layer_num].output)
        if not isinstance(conv_base.layers[layer_num], Dense) and not isinstance(conv_base.layers[layer_num], GlobalAveragePooling2D):
            intermediate_layer_model_padding_1 = tf.keras.Sequential([intermediate_layer_model,tf.keras.layers.ZeroPadding2D(padding=(1,1))])
        
        x_output = []
        image_num=-1
        
        max_val = 0
        image_num=0
        for item in imagenet_filelist:
            image_num=image_num+1
            if image_num<=1:
                # x_input = load_img(imagenet_path+'/'+item, target_size=(224,224)) #0.0~1.0, BGR
                # x_input = img_to_array(x_input)
                
                x_input = cv2.imread(imagenet_path+'/'+item)
                x_input = cv2.resize(x_input, (256, 256))
                # x_input = cv2.cvtColor(x_input, cv2.COLOR_BGR2RGB)
                
                mid_x, mid_y = 256//2, 256//2
                offset_x, offset_y = 224//2, 224//2
                
                x_input = x_input[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
            
                x_input = x_input[...,::-1].astype(np.float32)
                # x_input = np.asarray(x_input)
                            
    #               test=array_to_img(x_input)
                # x_input=x_input/1.0
    #                cv2.imshow("test",test)
                
                x_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1], x_input.shape[2])
                
                x_input = preprocess_input(x_input) #make 0.0~1.0 to -1.0~1.0?
                np.savetxt(os.path.join("dump2.txt"), np.array(x_input).transpose(0,3,1,2).flatten())
                x_output_temp= intermediate_layer_model(x_input)
                # x_output_temp= intermediate_layer_model.predict(x_input)
                print(np.max(np.abs(x_output_temp)))
                '''
                x_output_temp2=x_output_temp.flatten()
                i=0
                max_soft=0
                while i<x_output_temp2.size:
                    if max_soft<x_output_temp2[i]:
                        max_soft=x_output_temp2[i]
                    i+=1
                print(max_soft)
                '''
                
                if not isinstance(conv_base.layers[layer_num], Dense):
                    x_output_temp_pad_1=intermediate_layer_model_padding_1(x_input)
                    # x_output_temp_pad_1=intermediate_layer_model_padding_1.predict(x_input)
                    
                if image_num == 1:                                       
                    if isinstance(conv_base.layers[layer_num], Dense):
                        print(x_output_temp.shape)
                        np.savetxt(os.path.join(pathname, "dump_"+conv_base.layers[layer_num].name +"_activation.txt"), x_output_temp.numpy().flatten())
                        #np.savetxt(os.path.join(pathname, "dump_"+conv_base.layers[layer_num].name +"_activation_t.txt"), x_output_temp.transpose(1,0).flatten())
                        # print(x_output_temp.transpose(0,3,1,2).shape)
                        # print(x_output_temp.transpose(0,3,1,2)[0][0])
                    elif isinstance(conv_base.layers[layer_num], GlobalAveragePooling2D):
                        print(x_output_temp.shape)
                        np.savetxt(os.path.join(pathname, "dump_"+conv_base.layers[layer_num].name +"_activation.txt"), x_output_temp.numpy().flatten())
                        
                    else:
                        np.savetxt(os.path.join(pathname, "dump_"+conv_base.layers[layer_num].name +"_activation.txt"), x_output_temp.numpy().transpose(0,3,1,2).flatten())
                        #print(x_output_temp.transpose(0,3,1,2).shape)
                        np.savetxt(os.path.join(pathname, "dump_"+conv_base.layers[layer_num].name +"_activation_pad.txt"), x_output_temp_pad_1.numpy().transpose(0,3,1,2).flatten())
                        #print(x_output_temp_pad_1.transpose(0,3,1,2).shape)
                        #print(x_output_temp.transpose(0,3,1,2)[0][0])
                        #print(x_output_temp_pad_1.transpose(0,3,1,2)[0][0])
                    
                    
            else:
                break
       
