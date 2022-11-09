import argparse
from models import resnet50,resnet50_q, vgg16,vgg16_q, inception_v3, mobilenet, xception, squeezenet
from utils.load_weights import weight_loader
#from pkl_reader import DataGenerator
import tensorflow as tf
import cv2

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import os

test_image_num = 1000

'''
import numpy as np
test = np.array([130])
print(test)
test = test.astype(np.int8)
print(test)
'''

weights = {'vgg': 'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
           'vgg_q': 'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
           'resnet': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
           'resnet_q': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
           'inception': 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
           'xception': 'xception_weights_tf_dim_ordering_tf_kernels.h5',
           'squeezenet': 'squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
           'mobilenet_1.0': 'mobilenet_1_0_224_tf.h5',
           'mobilenet_0.75': 'mobilenet_7_5_224_tf.h5',
           'mobilenet_0.5': 'mobilenet_5_0_224_tf.h5',
           'mobilenet_0.25': 'mobilenet_2_5_224_tf.h5'
           }

def top5_acc(pred, k=5):
    Inf = 0.
    results = []
    for i in range(k):
        results.append(pred.index(max(pred)))
        pred[pred.index(max(pred))] = Inf
    return results


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='command for testing keras model with fp16 and fp32')
    parse.add_argument('--model', type=str, default='mobilenet', help='support vgg, resnet, densenet, \
         inception, inception_resnet, xception, mobilenet, squeezenet')
    parse.add_argument('--alpha', type=float, default=0.25, help='alpha for mobilenet')
    args = parse.parse_args()

    model_name = args.model if args.model != 'mobilenet' else args.model + '_' + str(args.alpha)
    weights = weight_loader('./weights/{}'.format(weights[model_name]))

    if args.model in ['vgg','vgg_q', 'resnet', 'resnet_q', 'mobilenet']:
        X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    elif args.model in ['inception', 'xception']:
        X = tf.placeholder(tf.float32, [None, 299, 299, 3])
    elif args.model == 'squeezenet':
        X = tf.placeholder(tf.float32, [None, 227, 227, 3])
    else:
        raise ValueError("Do not support {}".format(args.model))

    Y = tf.placeholder(tf.float32, [None, 1000])
    imagenet_path='../imagenet_valset'
    imagenet_num=5000
    imagenet_filelist = os.listdir(imagenet_path)
    imagenet_filelist.sort()

    
    
    #dg = DataGenerator('./data/val224_compressed.pkl', model=args.model, dtype='float32', num=test_image_num)
    with tf.device('/cpu:0'):
        if args.model == 'resnet':
            logits = resnet50.ResNet50(X, weights)
        elif args.model == 'resnet_q':
            logits = resnet50_q.ResNet50(X, weights)
            # logits_a = resnet50_q.TEST(X, weights)
        elif args.model == 'inception':
            logits = inception_v3.InceptionV3(X, weights)
        elif args.model == 'vgg_q':
            logits = vgg16_q.VGG16(X, weights)
            logits_a = vgg16_q.VGG16_a(X, weights)
        elif args.model == 'vgg':
            logits = vgg16.VGG16(X, weights)
        elif args.model == 'mobilenet':
            logits = mobilenet.MobileNet(X, weights, args.alpha)
        elif args.model == 'xception':
            logits = xception.Xception(X, weights)
        else:
            logits = squeezenet.SqueezeNet(X, weights)
        #prediction = tf.nn.softmax(logits)
        #pred = tf.argmax(prediction, 1)
        prediction = logits
        # dump = logits_a
        pred = tf.argmax(prediction, 1)
    # import numpy as np
    # from utils.layers import *
    # print(np.max(weights[block1_conv1_W_1:0]))
    acc = 0.
    acc_top5 = 0.
    
    f = open("../val.txt", 'r')
    print('Start evaluating {}'.format(args.model))
    with tf.Session() as sess:
        i=0
        # for im, label in dg.generator():
        for item in imagenet_filelist:
            #x_input = load_img(imagenet_path+'/'+item, target_size=(224,224)) #0.0~1.0, BGR
            #x_input = img_to_array(x_input)
            print(imagenet_path+'/'+item)
            x_input = cv2.imread(imagenet_path+'/'+item)
            # x_input = cv2.resize(x_input, (224, 224))
            x_input = cv2.resize(x_input, (256, 256))
            
            mid_x, mid_y = 256//2, 256//2
            offset_x, offset_y = 224//2, 224//2
       
            x_input = x_input[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
            # cv2.imshow('image', x_input)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            x_input = x_input[...,::-1].astype(np.float32)
            # x_input = x_input.astype(np.float32)
            
            x_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1], x_input.shape[2])
            print(x_input.shape)
            
            line = f.readline()
            line = line.strip()
            line = line.split(' ')[1]
            
            label = int(line)
            
            im = preprocess_input(x_input) #make 0.0~1.0 to -1.0~1.0?
            # np.savetxt(os.path.join("dump2.txt"), np.array(im).transpose(0,3,1,2).flatten())
            if i == -1:
                t1 = sess.run([dump], feed_dict={X: im})
                print("max:"+str(np.max(np.array(t1))))
                # np.savetxt(os.path.join("dump.txt"), np.array(t1).transpose(0,1,4,2,3).flatten())
                # print(np.array(t1))
                # np.savetxt(os.path.join("dump.txt"), np.array(t1).flatten())
            if  i<test_image_num:
                print('image'+str(i+1))
                t1, t5 = sess.run([pred, prediction], feed_dict={X: im})
                print(t1[0])
                if t1[0] == label:
                    acc += 1
                if label in top5_acc(t5[0].tolist()):
                    acc_top5 += 1
            else:
                break
            i+=1
            print(label)
            print('')
            print('Top1 accuracy: {}'.format(acc / i))
            print('Top5 accuracy: {}'.format(acc_top5 / i))

        print('Top1 accuracy: {}'.format(acc / test_image_num))
        print('Top5 accuracy: {}'.format(acc_top5 / test_image_num))
