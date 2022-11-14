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
#print("device 1:" + device_lib.list_local_devices()[1].name)

# netname="ResNet50"
# netname="ResNet50_Q"
# netname="VGG16"
netname="VGG16_Q"

quantname="INT8"
#quantname="INT16"
#quantname="INT32"
#quantname="FLOAT32"

WriteMode=True#Legacy
QuantizeDump=True

DumpLayerInfoMode=True
WeightQuantMode=True
ActivationQuantMode=True
DumpWeightMode=True

QuantizeTest=True
WaitMode=False


imagenet_path='imagenet_valset'
imagenet_num=5000
imagenet_filelist = os.listdir(imagenet_path)
imagenet_filelist.sort()
#초기값!
#C: 입력에 대한 채널
#K: 출력에 대한 체널
#num_RS is right?

num_WH=224;
num_RS=3;
ARRAY_C=4;

batch_size = 32
img_height = num_WH
img_width = num_WH
img_channel = ARRAY_C-1

pathname=netname+"_"+str(imagenet_num)+"/"

if netname in ['ResNet50','ResNet50_Q','ResNet50_Q']:
	from tensorflow.keras.applications.resnet50 import preprocess_input
	conv_base = ResNet50(weights='imagenet',include_top=True,input_shape=(224, 224, 3))
elif netname in ['VGG16','VGG16_Q']:
	from tensorflow.keras.applications.vgg16 import preprocess_input
	conv_base = VGG16(weights='imagenet',include_top=True,input_shape=(224, 224, 3))

#no quantization
if netname in ['ResNet50','VGG16']:
	QuantizeDump=False
	WeightQuantMode=False
	ActivationQuantMode=False
	
if quantname in ['INT8']:
	quant_max=127

#conv_base.summary()
conv_base.summary()
#데이터:CHW(RS)
#바이어스:K
#내부순서가중치:KCRS

if WaitMode is True:	input("Please press the Enter key to proceed")

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

createFolder(netname+"_"+str(imagenet_num))

#1:conv
#2:DepthwiseConv
#3:MaxPooling
#4:AvgPooling
#5:FC
#6:Mult
#7:Add
#8:GlAvgPool
#9:GlMaxPool
	 
	 
# Decode layer type as SPIE Layer Code
def decode_layer_type(layer, prev_layer=None, prev2_layer=None, prev3_layer=None, prev4_layer=None, next_layer=None, print_layer_info=False):
	#print(layer.name)
	if isinstance(layer, Conv2D): # Conv	
		RS = layer.kernel_size[0]
		Stride = layer.strides[0]
		prev_Padding = None
		if prev_layer is not None and isinstance(prev_layer, ZeroPadding2D):
			prev_Padding = prev_layer.padding
		if layer.padding == "same":
			Padding = RS // 2
		elif layer.padding == "valid":
			Padding = 0
			if prev_Padding is not None:
				Padding = min(prev_Padding[0][0], prev_Padding[0][1]) # min(top=left, bottom=right)
		#print("-RS=" + str(RS) + ", Stride=" + str(Stride) + "Padding=" + str(Padding))
		return (1,RS,Stride,Padding)

	elif isinstance(layer, DepthwiseConv2D): # Depthwise Conv
		RS = layer.kernel_size[0]
		Stride = layer.strides[0]
		prev_Padding = None
		if prev_layer is not None and isinstance(prev_layer, ZeroPadding2D):
			prev_Padding = prev_layer.padding
		if layer.padding == "same":
			Padding = RS // 2
		elif layer.padding == "valid":
			Padding = 0
			if prev_Padding is not None:
				Padding = min(prev_Padding[0][0], prev_Padding[0][1]) # min(top=left, bottom=right)
		#print("-RS=" + RS,", Stride=" + S + "Padding=" + P)
		return (2,RS,Stride,Padding)

	elif isinstance(layer, MaxPooling2D): # MaxPooling			
		RS = layer.pool_size[0]
		Stride = layer.strides[0]
		prev_Padding = None
		if prev_layer is not None and isinstance(prev_layer, ZeroPadding2D):
			prev_Padding = prev_layer.padding
		if layer.padding == "same":
			Padding = RS // 2
		elif layer.padding == "valid":
			Padding = 0
			if prev_Padding is not None:
				Padding = min(prev_Padding[0][0], prev_Padding[0][1])
		if print_layer_info: print("Layer Type : Local Max Pool")
		#print("-RS=" + str(RS) + ", Stride=" + str(Stride) + "Padding=" + str(Padding))
		return (3,RS,Stride,Padding)

	elif isinstance(layer, AveragePooling2D): # AveragePooling  
		RS = layer.pool_size[0]
		Stride = layer.strides[0]
		prev_Padding = None
		if prev_layer is not None and isinstance(prev_layer, ZeroPadding2D):
			prev_Padding = prev_layer.padding
		if layer.padding == "same":
			Padding = RS // 2
		elif layer.padding == "valid":
			Padding = 0
			if prev_Padding is not None:
				P = min(prev_Padding[0][0], prev_P[0][1])
		if print_layer_info: print("Layer Type : Local Max Pool")
		#print("-RS=" + str(RS) + ", Stride=" + str(Stride) + "Padding=" + str(Padding))
		return (4,RS,Stride,Padding)
		
	elif isinstance(layer, Dense): # Dense	   
		return (5,0,0,0)
		
	elif isinstance(layer, Multiply): # Mult
		if layer.input[0].name==prev4_layer.output.name or layer.input[1].name==prev4_layer.output.name:  # H Swish
			return (None,0,0,0)
		return (6,0,0,0)
		
	elif isinstance(layer, Add): # Add
		return (7,0,0,0)
		
	elif isinstance(layer, GlobalAveragePooling2D): # Global AvgPooling
		return (8,0,0,0)

	elif isinstance(layer, GlobalMaxPooling2D): # Global MaxPooling
		return (9,0,0,0)
	   
	else:
		return (None,0,0,0)


def getLayerIndex(model, layer):
	layer_names = [layer.name for layer in model.layers]
	layer_idx = layer_names.index(layer.name)
	return layer_idx
	
def getTensorIndex(model, tensor):
	if tensor.name.rfind("/")==-1:
		layer_name = tensor.name
		if tensor.name.rfind(":")==-1:
			layer_name = tensor.name
		else: # For ResNet Layer:  input_1:0
			layer_name = tensor.name[:tensor.name.rfind(":")]
	else:
		layer_name = tensor.name[:tensor.name.rfind("/")]
	if layer_name[-5:]=="/cond":
		layer_name = layer_name[:-5]

	layer_name = tensor.name
	tensor_names = [layer.output.name for layer in model.layers]
	layer_idx = tensor_names.index(layer_name)
	return layer_idx

# Specify layer inputs and activation
def specify_layer_info(model, inherit_tensors, layer, prev_layer=None, prev2_layer=None, prev3_layer=None, prev4_layer=None, next_layer=None, print_layer_info=False):
	#print(layer.name)
	layer_num = getLayerIndex(model, layer)
	input_num = []
	if isinstance(layer.input, list):
		for input_tensor in layer.input:
			input_num.append(inherit_tensors[getTensorIndex(model, input_tensor)])
	else:
		input_num.append(inherit_tensors[getTensorIndex(model, layer.input)])
	print(layer_num, " ", input_num)

	print(layer.name)
	if isinstance(layer, DepthwiseConv2D): # Depthwise Conv
		inherit_tensors[layer_num] = layer_num
		return (layer_num, input_num, None)
		
	elif isinstance(layer, Conv2D): # Conv
		inherit_tensors[layer_num] = layer_num
		return (layer_num, input_num, None)
		
	elif isinstance(layer, Dense): # Dense
		inherit_tensors[layer_num] = layer_num
		return (layer_num, input_num, None)
		
	elif isinstance(layer, ReLU):  # ReLU
		inherit_tensors[layer_num] = inherit_tensors[input_num[0]]
		return (None, None, "ReLU")
		
	elif isinstance(layer, TFOpLambda) and isinstance(prev_layer, ReLU):  # Assume, Sigmoid
		print("AAAAAAAAA")
		inherit_tensors[layer_num] = inherit_tensors[input_num[0]]
		return (None, None, "H-Sigmoid")
		
	elif isinstance(layer, Multiply): # Mult
		if layer.input[0].name==prev4_layer.output.name or layer.input[1].name==prev4_layer.output.name:  # H Swish
			if inherit_tensors[input_num[0]] != inherit_tensors[input_num[1]]:
				raise Exception("Something error in input")
			inherit_tensors[layer_num] = inherit_tensors[input_num[0]]
			return (None, None, "H-Swish")
		inherit_tensors[layer_num] = layer_num
		return (layer_num, input_num, None)
		
	elif isinstance(layer, Add): # Add
		inherit_tensors[layer_num] = layer_num
		return (layer_num, input_num, None)
		
	elif isinstance(layer, GlobalAveragePooling2D): # Global AvgPooling
		inherit_tensors[layer_num] = layer_num
		return (layer_num, input_num, None)
		
	elif isinstance(layer, MaxPooling2D): # MaxPooling
		inherit_tensors[layer_num] = layer_num
		return (layer_num, input_num, None)
	else:
		if len(input_num) > 1:
			raise Exception("Input Number is not 1")
		inherit_tensors[layer_num] = inherit_tensors[input_num[0]]
		return None
		
if DumpLayerInfoMode is True:
	four_last_l = None
	three_last_l = None
	two_last_l = None
	last_l = None
	inherit_tensors = []
	inherit_tensors_spie = []
	layer_infos = []
	next_threshold_num = 0 #pointer of threshold for next threshold layer
	threshold_infos = []
	for l in conv_base.layers:
		# print_layer_info(l)
		print(" ")
		print("Add Layer "+ str(getLayerIndex(conv_base, l))+":"+ l.name)
		inherit_tensors.append(getLayerIndex(conv_base, l))
		inherit_tensors_spie.append(None)
		
		type = decode_layer_type(l, prev_layer=last_l, prev2_layer=two_last_l, prev3_layer=three_last_l, prev4_layer=four_last_l)   
		Linfo = specify_layer_info(conv_base, inherit_tensors, l, prev_layer=last_l, prev2_layer=two_last_l, prev3_layer=three_last_l, prev4_layer=four_last_l)


		if type[0] != None: # Valid SPIE Layer
			if Linfo[0] == None: # Invalid Info
				raise Exception("Invalid SPIE Layer")

			ch_in = conv_base.get_layer(index=Linfo[1][0]).output.shape[-1]
			if len(Linfo[1])>1: # Add, Mul, ...
				if len(conv_base.get_layer(index=Linfo[1][0]).output.shape) == 4:
					dim_in = max(conv_base.get_layer(index=Linfo[1][0]).output.shape[1], conv_base.get_layer(index=Linfo[1][1]).output.shape[1])
				elif len(conv_base.get_layer(index=Linfo[1][0]).output.shape) == 2:
					dim_in = 1
			else:
				if len(conv_base.get_layer(index=Linfo[1][0]).output.shape) == 4:
					dim_in = conv_base.get_layer(index=Linfo[1][0]).output.shape[1]
				elif len(conv_base.get_layer(index=Linfo[1][0]).output.shape) == 2:
					dim_in = 1
			ch_out = conv_base.get_layer(index=Linfo[0]).output.shape[-1]
			if len(conv_base.get_layer(index=Linfo[0]).output.shape) == 4:
				dim_out = conv_base.get_layer(index=Linfo[0]).output.shape[1]
			elif len(conv_base.get_layer(index=Linfo[0]).output.shape) == 2:
				dim_out = 1

			in1 = inherit_tensors_spie[Linfo[1][0]]
			if len(Linfo[1]) == 2:
				in2 = inherit_tensors_spie[Linfo[1][1]]
			else:
				in2 = -2
			if in1 == None:
				in1 = -1
			if in2 == None:
				in2 = -2
			
			act=0;

			inherit_tensors_spie[getLayerIndex(conv_base, l)] = len(layer_infos)
			layer_infos.append([l.name, type[0], type[1], type[2], type[3], ch_in, dim_in, ch_out, dim_out, in1, in2, act])
		else:
			#act_fix
			if Linfo is not None and Linfo[2] is not None: # Activation, etc, ..
				if Linfo[2] == "ReLU":
					layer_infos[-1][11] = 1
						#ReLU6?
				elif Linfo[2] == "H-Sigmoid":
					layer_infos[-1][11] = 3
				elif Linfo[2] == "H-Swish":
					layer_infos[-1][11] = 4
		#threshold
		if "percentage" in l.name and threshold_file is not None:
			npnt = next_threshold_num
			if threshold_content[2][npnt] in l.name:
				threshold_infos.append([len(layer_infos), l.name, threshold_content[4][npnt], threshold_content[5][npnt], threshold_content[6][npnt], 0]) #threshold file format, 0=input sparsification
				next_threshold_num += 1
		four_last_l = three_last_l
		three_last_l = two_last_l
		two_last_l = last_l
		last_l = l

	print(netname)
	structure = open(pathname+"structure.csv","w")
	namemap = open(pathname+"name_table","w")
	structure.write("{0}\n".format(len(layer_infos)))

	print('					: layertype ,  RS ,Strid,Padin,  C  ,WH_in,  K  ,WH_ou, in1 , in2 , act ')
	for l in layer_infos:
		print('%20s: %10s, %4s, %4s, %4s, %4s, %4s, %4s, %4s, %4s, %4s, %4s' % (l[0],l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9],l[10],l[11]))
		structure.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n".format(l[1],l[2],l[3],l[4],l[5],l[6],l[7],l[8],l[9],l[10],l[11]))

	for l in layer_infos:
		lname = l[0]
		lname = lname.replace("/","_")
		#print(lname)
		namemap.write(lname+"\n")
	namemap.write("\n")
	structure.close()
	namemap.close()
	if WaitMode is True:	input("Please press the Enter key to proceed")


import cv2
		
layer_names_qaunt = []
conv_layers_qaunt = []
layer_infos_weight_qaunt = []


def computeWeightQuantize(array):
	max_val = 0
	scale_factor=0
	i=0
	print("quantization size: "+str(array.size))
	max_val = np.max(np.abs(array))
	print("maxval:"+str(max_val))
	if max_val < 0.0001:
		scale_factor = 0
	else:
		scale_factor = quant_max / max_val
	return scale_factor
	
def bn_folding_quantize (conv_weights, bn_weights, network_name, path_name, layer_name):
	if conv_weights[0].shape[-1] == 1 :
		conv_weight = np.squeeze(conv_weights[0].numpy())
	else :
		conv_weight = conv_weights[0].numpy()
	gamma = bn_weights[0].numpy()
	beta = bn_weights[1].numpy()
	mean = bn_weights[2].numpy()
	squared_var = bn_weights[3].numpy()
	if network_name == 'ResNet50':
		print("ResNet50 eps = 1.001e-5")
		eps = 1.001e-5
	else :
		print("MobileNet eps = 1e-3")
		eps = 1e-3

	fused_weight = np.multiply(conv_weight,gamma) / np.sqrt(squared_var + eps)
	#fused_weight = fused_weight.transpose(3,2,0,1)
	if len(fused_weight.shape) == 3:
		fused_weight = fused_weight.transpose(2,0,1).flatten()
	else :
		fused_weight = fused_weight.transpose(3,2,0,1).flatten()
 
	#np.savetxt(os.path.join(path_name, layer_name + "_weight.txt"), fused_weight)
	if len(conv_weights) > 1 :
		conv_bias = conv_weights[1].numpy()
	else :
		conv_bias = 0
	fused_bias = (np.multiply(np.subtract(conv_bias,mean) ,gamma) / np.sqrt(squared_var +eps)) + beta
	#np.savetxt(os.path.join(path_name, layer_name + "_bias.txt"), fused_bias)

	weight_quant=computeWeightQuantize(fused_weight)
	bias_quant  =computeWeightQuantize(fused_bias)
	print(layer_name +" weight scale factor: " + str(weight_quant))
	print(layer_name +" bias scale factor: " + str(bias_quant))
	layer_infos_weight_qaunt.append([conv.name,weight_quant,bias_quant])
  

if WeightQuantMode is True:
	print("Weight quantization")
	print("====================================")
	print(" Dumping Layer Weight And Bias")
	for l in conv_base.layers:
		#conv layer(import later!)
		if isinstance(l, Conv2D):
			layer_names_qaunt.append(l.name)
			conv_layers_qaunt.append(l)
		if isinstance(l, DepthwiseConv2D):
			layer_names_qaunt.append(l.name)
			conv_layers_qaunt.append(l)
		if isinstance(l, BatchNormalization):
			layer_names_qaunt.append(l.name)
		if isinstance(l, Dense):
			layer_names_qaunt.append(l.name)
			conv_layers_qaunt.append(l)
			
		#개수 맞추기 위한 용도
		if isinstance(l, Multiply):
			conv_layers_qaunt.append(l)
		if isinstance(l, Add):
			conv_layers_qaunt.append(l)
		if isinstance(l, GlobalAveragePooling2D):
			conv_layers_qaunt.append(l)
		if isinstance(l, GlobalMaxPooling2D):
			conv_layers_qaunt.append(l)
		if isinstance(l, AveragePooling2D):
			conv_layers_qaunt.append(l)
		if isinstance(l, MaxPooling2D):
			conv_layers_qaunt.append(l)
			
	#print(layer_names_qaunt)		
	for conv in conv_layers_qaunt :
	#for conv in conv_base.layers:	
		print("====================================")
		if conv.name.find("_conv") !=-1:
			if netname in ['ResNet50','ResNet50_Q','ResNet101','ResNet152','ResNet50V2','ResNet101V2','ResNet152V2']:
				bn_name = conv.name.replace("_conv","_bn")
			else:
				bn_name = conv.name + '/BatchNorm'
		else:
			bn_name = "write_some_long_letter_so_that_program_fucked_up"
			
		conv_weights = conv.weights
		
		#개수 맞추기 위한 용도
		if isinstance(conv, Multiply):
			print(conv.name+ " has no weight")
			layer_name = conv.name.replace("/","_")
			layer_infos_weight_qaunt.append([layer_name,0,0])
		elif isinstance(conv, Add):
			print(conv.name+ " has no weight")
			layer_name = conv.name.replace("/","_")
			layer_infos_weight_qaunt.append([layer_name,0,0])
		elif isinstance(conv, GlobalAveragePooling2D):
			print(conv.name+ " has no weight")
			layer_name = conv.name.replace("/","_")
			layer_infos_weight_qaunt.append([layer_name,0,0])
		elif isinstance(conv, GlobalMaxPooling2D):
			print(conv.name+ " has no weight")
			layer_name = conv.name.replace("/","_")
			layer_infos_weight_qaunt.append([layer_name,0,0])
		elif isinstance(conv, AveragePooling2D):
			print(conv.name+ " has no weight")
			layer_name = conv.name.replace("/","_")
			layer_infos_weight_qaunt.append([layer_name,0,0])
		elif isinstance(conv, MaxPooling2D):
			print(conv.name+ " has no weight")
			layer_name = conv.name.replace("/","_")
			layer_infos_weight_qaunt.append([layer_name,0,0])

		elif bn_name in layer_names_qaunt:
			print("caculating " + conv.name + " + " + bn_name+ " layer")
			bn_layer = conv_base.get_layer(bn_name)
			bn_weights = bn_layer.weights
			layer_name = conv.name.replace("/","_")
			bn_folding_quantize(conv_weights, bn_weights, netname, pathname, layer_name)
		elif conv.name in layer_names_qaunt:
		
			if isinstance(conv , Dense):
				print("caculating " + conv.name + " layer")
				layer_name = conv.name.replace("/","_")
				conv_weight = conv_weights[0].numpy().transpose(1,0).flatten()
				if len(conv_weights) > 1 :
					conv_bias = conv_weights[1].numpy()
				else:
					conv_bias=[0]

			else:
				print("caculating " + conv.name + " layer")
				layer_name = conv.name.replace("/","_")
				conv_weight = conv_weights[0].numpy().transpose(3,2,0,1).flatten()
				if len(conv_weights) > 1 :
					conv_bias = conv_weights[1].numpy()
				else:
					conv_bias=[0]				   
			
			weight_quant=computeWeightQuantize(conv_weight)
			bias_quant  =computeWeightQuantize(conv_bias)
			print(layer_name +" weight scale factor: " + str(weight_quant))
			print(layer_name +" bias scale factor: " + str(bias_quant))
			layer_infos_weight_qaunt.append([layer_name,weight_quant,bias_quant])
	
	weight_qaunt_list = open(pathname+"weight_quant","w")	
	print('%20s: %20s, %20s' % ("layername","weight_scale","bias_scale(f"))
	for l in layer_infos_weight_qaunt:   
		print('%20s: %20s, %20s' % (l[0],l[1],l[2]))
		weight_qaunt_list.write("{0},{1}\n".format(l[1],l[2]))

	print("%s weight quantization done !!! " %netname)
	weight_qaunt_list.close()
	if WaitMode is True:	input("Please press the Enter key to proceed")


#kl = tf.keras.losses.KLDivergence()


if ActivationQuantMode is True:
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
		if netname in ['ResNet50','ResNet50_Q','ResNet50_Q','ResNet101','ResNet152','ResNet50V2','ResNet101V2','ResNet152V2']:
			#if isinstance(l, InputLayer):
			#	print("skip")
			#	skip_bool=True
			if isinstance(l, ZeroPadding2D):
				skip_bool=True
			if isinstance(l, Flatten):
				skip_bool=True
			if isinstance(l, Add):
				skip_bool=True
			# elif isinstance(l, ReLU):
				# skip_bool=True
			if l.name.find("_conv") !=-1:		   
				temp_str=l.name.replace("_conv","_bn")#batch_normalization
				temp_str2=l.name.replace("_conv","_relu")#activation
				for ll in conv_base.layers:
					if ll.name in [temp_str,temp_str2]:
					# if ll.name in [temp_str]:
						skip_bool=True			   
			elif l.name.find("_bn") !=-1:
				temp_str=l.name.replace("_bn","_relu")#activation
				for ll in conv_base.layers:
					if ll.name in [temp_str]:
						skip_bool=True		 
			#elif l.name.find("_relu") !=-1:
			#what
			
			elif l.name.find("_add") !=-1:
				temp_str=l.name.replace("_add","_out")#activation
				for ll in conv_base.layers:
					if ll.name in [temp_str]:
						skip_bool=True
		
		elif netname in ['VGG16', 'VGG16_Q']:
			if isinstance(l, Flatten):
				skip_bool=True		
		if skip_bool is True:
			print("skip")
			continue

		intermediate_layer_model=tf.keras.Model(inputs=conv_base.input, outputs=conv_base.layers[layer_num].output)
		layer_name = conv_base.layers[layer_num].name.replace("/","_")
		#x_output_temp = [0] * imagenet_num
		x_output = []
		image_num=-1
		
		max_val = 0
		image_num=0
		for item in imagenet_filelist:
			image_num=image_num+1
			if image_num<=imagenet_num:
				if image_num%100 == 0 or imagenet_num<100:
					print('Load Image:' + imagenet_path + '/' + item)
				   #TODO: FIX RIGHTLY
				x_input = load_img(imagenet_path+'/'+item, target_size=(224,224)) #0.0~1.0, BGR
				x_input = img_to_array(x_input)
				# x_input = cv2.imread(imagenet_path+'/'+item)
				# x_input = cv2.resize(x_input, (256, 256))
			
				# mid_x, mid_y = 256//2, 256//2
				# offset_x, offset_y = 224//2, 224//2
	   
				# x_input = x_input[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
			
				# x_input = x_input[...,::-1].astype(np.float32)
				
				x_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1], x_input.shape[2])
				#x_input = input_image_preprocess(x_input, mode='caffe')
				#x_input = input_image_preprocess(x_input, mode='tf')
				
				x_input = preprocess_input(x_input) #make 0.0~1.0 to -1.0~1.0?
				# if image_num == 1:									   
					# np.savetxt(os.path.join(pathname, conv_base.layers[layer_num].name +"_activation.txt"), x_output_temp.numpy().transpose(0,3,1,2).flatten())
					# print(x_output_temp.numpy().transpose(0,3,1,2).shape)
					# np.savetxt(os.path.join(pathname, conv_base.layers[layer_num].name +"_activation_t.txt"), x_output_temp.numpy().transpose(0,3,2,1).flatten())   
					# print(x_output_temp.numpy().transpose(0,3,2,1).shape)
				# x_output_temp= intermediate_layer_model(x_input)
				# x_output_temp=x_output_temp.numpy().flatten()
				x_output_temp= intermediate_layer_model.predict(x_input)
				x_output_temp=x_output_temp.flatten()
				
				temp_max = np.max(np.abs(x_output_temp))
				if max_val < temp_max:
					max_val = temp_max
				del x_input
				del x_output_temp
			else:
				break
				
		print("max:" + str(max_val))
		histogram = np.array([0]*quantize_level)
		
		image_num=0
		total_value_count=0
		for item in imagenet_filelist:
			image_num=image_num+1
			if image_num<=imagenet_num:
				if image_num%100 == 0 or imagenet_num<100:
					print('Load Image:' + imagenet_path + '/' + item)
				# x_input = load_img(imagenet_path+'/'+item, target_size=(224,224)) #0.0~1.0, BGR
				# x_input = img_to_array(x_input) #imag->numpy
				
				# test=array_to_img(x_input)
				# test=test/256.0
				# cv2.imshow("test",test)

				#(H, W, C), BGR format, float(0~255)
				x_input = load_img(imagenet_path+'/'+item, target_size=(224,224)) #0.0~1.0, BGR
				x_input = img_to_array(x_input)
				# x_input = cv2.imread(imagenet_path+'/'+item)
				# x_input = cv2.resize(x_input, (256, 256))
			
				# mid_x, mid_y = 256//2, 256//2
				# offset_x, offset_y = 224//2, 224//2
	   
				# x_input = x_input[mid_y - offset_y:mid_y + offset_y, mid_x - offset_x:mid_x + offset_x]
			
				# x_input = x_input[...,::-1].astype(np.float32)
				
				x_input = x_input.reshape(1, x_input.shape[0], x_input.shape[1], x_input.shape[2])
							  # x_output_temp= intermediate_layer_model(x_input)
				# x_output_temp=x_output_temp.numpy().flatten()
				x_output_temp= intermediate_layer_model.predict(x_input)
				x_output_temp=np.abs(x_output_temp.flatten())
				
				# x_output_temp= intermediate_layer_model(x_input)
				# x_output_temp=np.abs(x_output_temp.numpy().flatten())
				
				# x_output_temp=np.abs(x_output_temp.numpy())
				#x_output_temp=np.where(x_output_temp < 0.001, -2, (x_output_temp/max_val)*(quantize_level-1))
				#x_output_temp=np.round(x_output_temp)
				#x_output_temp=x_output_temp.astype(np.int32)
				i=0
				
				histogram_temp=0 #요소가 4096개이면 4097구간으로 쪼개야함
				hist,bins = np.histogram(x_output_temp,bins=quantize_level,range=[-max_val/(quantize_level*2-1),max_val])
				# print(hist.size)
				# print(hist)
				# print(bins)
				histogram+=hist
				total_value_count +=hist.size
				
				del x_input
				del x_output_temp
			else:
				break
				
		#total_value_count: 총 개수 
			
		original_descale_factor=max_val/(quantize_level-1)
		original_sum=0
		i=0	
		while i < quantize_level:
			original_sum=histogram[i]
			i=i+1   
			
		cutting_descale_factor=original_descale_factor
		cutting_sum=0
		fixed_sum=0
		cutting=quantize_level-1
		divergence_good=999999
		divergence_good_cutting=-1
		good_scale_factor=0
		
		divergence_temp_sum_2=0
		while i < quantize_level:
		# +p_i*log(p_i)
			temp_k=histogram[i]
			if temp_k==0:
				temp_k=0.0001
			divergence_temp_sum_2+=temp_k*math.log(temp_k)
			i+=1
		
		while cutting >= 1:
			#cutting_descale_factor=original_descale_factor*quantize_level/cutting
			divergence_val=0
			i=0
			#P: 이상(원본값)
			#Q: 현실
			
			divergence_val_temp_sum=divergence_temp_sum_2;
			while i < cutting:
			# -p_i*log(q_i)
				temp_k=histogram[i]
				if temp_k==0:
					temp_k=0.0001
				divergence_val-=temp_k*math.log(temp_k)
				#divergence_val_temp_sum+=histogram[i] # <-이거 뭐지?
				i+=1
			# i == cutting
			temp_k=total_value_count-divergence_val_temp_sum
			if temp_k==0:
				temp_k=0.0001
			divergence_val-=histogram[i]*math.log(temp_k)
			
			if divergence_val < divergence_good:
				divergence_good = divergence_val
				divergence_good_cutting = cutting
				good_scale_factor =  quant_max / (divergence_good_cutting*original_descale_factor)#cutting*cutting_descale_factor에서 딱 자르겠다 -> 이값을 127로 만들겠다 
			cutting = cutting - 1
		print("cutting max: " +str(original_descale_factor*divergence_good_cutting))
		print(str(divergence_good_cutting) + " : " +str(divergence_good))
		
		print(str(good_scale_factor))
		
		filename_layer_name = layer_name
		if netname in ['ResNet50','ResNet50_Q','ResNet50_Q','ResNet101','ResNet152','ResNet50V2','ResNet101V2','ResNet152V2']:
			if filename_layer_name.find("_bn") !=-1:		   
				filename_layer_name=filename_layer_name.replace("_bn","_conv")
			elif filename_layer_name.find("_relu") !=-1:		   
				filename_layer_name=filename_layer_name.replace("_relu","_conv")
			elif filename_layer_name.find("_out") !=-1:
				filename_layer_name=filename_layer_name.replace("_out","_add")
		
		layer_infos_active_quant.append([filename_layer_name, good_scale_factor, divergence_good_cutting])
		#np.savetxt(os.path.join(pathname, filename_layer_name +"_activation.txt"), x_output)	
		del x_output
		del intermediate_layer_model
		#intermediate_output=intermediate_layer_model2(x_input)
				#intermediate_layer_model[layer_num].summary()
	active_qaunt_list = open(pathname+"active_quant","w")	
	print('%20s: %20s, %20s' % ("layername","active_out_scale","chonese_cutting"))
	for l in layer_infos_active_quant:   
		print('%20s: %20s, %20s' % (l[0],l[1],l[2]))
		active_qaunt_list.write("{0}\n".format(l[1]))
	active_qaunt_list.close()	
		
	print("====================================")
	
	if WaitMode is True:	input("Please press the Enter key to proceed") 


def ArrayQuantize(weights, scale):
	#우리가 주는 값은 곱해야함...
	# qweights = weights / s
	qweights = weights * scale
	qweights = np.round(qweights)
	#CLAMP
	qweights = np.where(qweights > 127, 127, qweights)
	qweights = np.where(qweights < -127, -127, qweights)
	qweights = qweights.astype(np.int8)
	return qweights

def ArrayQuantize_MAC(weights, scale):
	#우리가 주는 값은 곱해야함...
	# qweights = weights / s
	qweights = weights * scale
	qweights = np.round(qweights)
	#CLAMP
	qweights = np.where(qweights > 2147483647, 2147483647, qweights)
	qweights = np.where(qweights < -2147483647, -2147483647, qweights)
	qweights = qweights.astype(np.int32)
	return qweights


def getLayerInfoIndex(list, name):
	j=0
	for i in list:
		if i[0] in name:
			return j
		j=j+1
	print("Error")
	return None

# For BN Folding of Conv Layers
def bn_folding (conv_weights, bn_weights, network_name, path_name, layer_name):
	if conv_weights[0].shape[-1] == 1 :
		conv_weight = np.squeeze(conv_weights[0].numpy())
	else :
		conv_weight = conv_weights[0].numpy()
	gamma = bn_weights[0].numpy()
	beta = bn_weights[1].numpy()
	mean = bn_weights[2].numpy()
	squared_var = bn_weights[3].numpy()
	if network_name == 'ResNet50':
		print("ResNet50 eps = 1.001e-5")
		eps = 1.001e-5
	else :
		print("MobileNet eps = 1e-3")
		eps = 1e-3

	fused_weight = np.multiply(conv_weight,gamma) / np.sqrt(squared_var + eps)
	#fused_weight = fused_weight.transpose(3,2,0,1)
	if len(fused_weight.shape) == 3:
		fused_weight = fused_weight.transpose(2,0,1).flatten()
	else :
		fused_weight = fused_weight.transpose(3,2,0,1).flatten()
	if QuantizeDump is True:
		index=getLayerInfoIndex(layer_infos_weight_qaunt,layer_name)
		weight_quant=layer_infos_weight_qaunt[index][1]
		print("Caculated Weig Quant : " + str(weight_quant))
		fused_weight=ArrayQuantize(fused_weight,weight_quant)
	np.savetxt(os.path.join(path_name, layer_name + "_weight.txt"), fused_weight)
	print(layer_name +" weight size: " + str(conv_weight.size))
	print("BNFold Conv weight saved: ", os.path.join(path_name, layer_name + "_weight.txt"))
	if len(conv_weights) > 1 :
		conv_bias = conv_weights[1].numpy()
	else :
		conv_bias = 0
	fused_bias = (np.multiply(np.subtract(conv_bias,mean) ,gamma) / np.sqrt(squared_var +eps)) + beta
	if QuantizeDump is True:
		index=getLayerInfoIndex(layer_infos,layer_name)
		newindex=layer_infos[index][9]+1#9번:in1
		if layer_infos_active_quant[newindex][0].find("_pool") !=-1: 
			rnewindex=layer_infos[index][9]
			newindex=layer_infos[rnewindex][9]+1#9번:in1
		print("get from " + str(layer_infos_active_quant[newindex][0]))
		activation_quant=layer_infos_active_quant[newindex][1]
		fused_bias=ArrayQuantize_MAC(fused_bias,weight_quant*activation_quant) 
		print("Caculated Bias Quant : " + str(layer_infos_weight_qaunt[index][2]))
		print("Real(smal Bias Quant : " + str(weight_quant*activation_quant))
	np.savetxt(os.path.join(path_name, layer_name + "_bias.txt"), fused_bias)
	print(layer_name +" bias   size: " + str(fused_bias.size))
	print("BNFold Conv bias saved: ", os.path.join(path_name, layer_name + "_bias.txt"))	

# Split Conv Layers & Save FC Layer Weight'
if DumpWeightMode is True:
	layer_infos_quant = []
	print("====================================")
	print(" Dumping Layer Weight And Bias")
	for l in conv_base.layers:
		#conv layer(import later!)
		if isinstance(l, Conv2D):
			conv_layers.append(l)
			layer_names.append(l.name)
		if isinstance(l, DepthwiseConv2D):
			conv_layers.append(l)
			layer_names.append(l.name)
		if isinstance(l, BatchNormalization):
			layer_names.append(l.name)
		#fc layer dump
		
		if isinstance(l , Dense):
			print("====================================")
			print("dumping " + l.name + " layer")
			#weights[0] [C*W*H][K]
			print(l.weights[0].numpy().transpose(1,0).shape)
			#transors [K][C*W*H] -> [
			fc_weight = l.weights[0].numpy().transpose(1,0).flatten()
			layer_name = l.name.replace("/","_")
			if QuantizeDump is True:
				index=getLayerInfoIndex(layer_infos_weight_qaunt,layer_name)
				weight_quant=layer_infos_weight_qaunt[index][1]
				print("Caculated Weig Quant : " + str(weight_quant))
				fc_weight=ArrayQuantize(fc_weight,weight_quant)
			np.savetxt(os.path.join(pathname, layer_name + "_weight.txt"), fc_weight)
			print(layer_name +" weight size: " + str(fc_weight.size))
			print("FC weight saved: ", os.path.join(pathname, layer_name + "_weight.txt"))
			if len(l.weights) > 1 :
				fc_bias = l.weights[1].numpy()
				if QuantizeDump is True:
					#index=getLayerInfoIndex(layer_infos,layer_name)
					newindex=layer_infos[index][9]+1#9번:in1
					if layer_infos_active_quant[newindex][0].find("_pool") !=-1: 
						rnewindex=layer_infos[index][9]
						newindex=layer_infos[rnewindex][9]+1#9번:in1
					print("get from " + str(layer_infos_active_quant[newindex][0]))
					activation_quant=layer_infos_active_quant[newindex][1]
					fc_bias=ArrayQuantize_MAC(fc_bias,weight_quant*activation_quant)
					print("Caculated Bias Quant : " + str(layer_infos_weight_qaunt[index][2]))
					print("Real(smal Bias Quant : " + str(weight_quant*activation_quant))
				np.savetxt(os.path.join(pathname, layer_name + "_bias.txt"), fc_bias)
				print(layer_name +" bias   size: " + str(fc_bias.size))
				print("FC bias saved: ", os.path.join(pathname, layer_name + "_bias.txt"))
		
	
	#print(layer_names)
	#import layer
	for conv in conv_layers :
		print("====================================")
		if conv.name.find("_conv") !=-1:
			if netname in ['ResNet50','ResNet50_Q','ResNet101','ResNet152','ResNet50V2','ResNet101V2','ResNet152V2']:
				bn_name = conv.name.replace("_conv","_bn")
			else:
				bn_name = conv.name + '/BatchNorm'
		else:
			bn_name = "write_some_long_letter_so_that_program_fucked_up"
		
		conv_weights = conv.weights
		
		if bn_name in layer_names:
			print("dumping " + conv.name + " + " + bn_name+ " layer")
			bn_layer = conv_base.get_layer(bn_name)
			bn_weights = bn_layer.weights
			layer_name = conv.name.replace("/","_")
			bn_folding(conv_weights, bn_weights, netname, pathname, layer_name)
		else :
			print("dumping " + conv.name + " layer")
			layer_name = conv.name.replace("/","_")
			conv_weight = conv_weights[0].numpy().transpose(3,2,0,1).flatten()
			
			print(conv_weights[0].numpy().transpose(3,2,0,1).shape)
			print(conv_weights[0].numpy().transpose(3,2,0,1)[1][0])
			#[K][C][R][S]
			print(layer_name +" weight size: " + str(conv_weight.size))
			if QuantizeDump is True:
				index=getLayerInfoIndex(layer_infos_weight_qaunt,layer_name)
				weight_quant=layer_infos_weight_qaunt[index][1]
				print("Caculated Weig Quant : " + str(weight_quant))
				conv_weight=ArrayQuantize(conv_weight,weight_quant)
			np.savetxt(os.path.join(pathname, layer_name +"_weight.txt"), conv_weight)
			print("Normal Conv weight saved: ", os.path.join(pathname, layer_name +"_weight.txt"))
			if len(conv_weights) >1 :
				conv_bias = conv_weights[1].numpy()
				print(conv_bias)
				print(layer_name +" bias   size: " + str(conv_bias.size))
				if QuantizeDump is True:
					index=getLayerInfoIndex(layer_infos,layer_name)
					newindex=layer_infos[index][9]+1#9번:in1
					if layer_infos_active_quant[newindex][0].find("_pool") !=-1: 
						rnewindex=layer_infos[index][9]
						newindex=layer_infos[rnewindex][9]+1#9번:in1
					print("get from " + str(layer_infos_active_quant[newindex][0]))
					activation_quant=layer_infos_active_quant[newindex][1]
					conv_bias=ArrayQuantize_MAC(conv_bias,weight_quant*activation_quant) 
					print("Caculated Bias Quant : " + str(layer_infos_weight_qaunt[index][2]))
					print("Real(smal Bias Quant : " + str(weight_quant*activation_quant))
				np.savetxt(os.path.join(pathname, layer_name +"_bias.txt"), conv_bias)
				print("Normal Conv bias saved: ", os.path.join(pathname, layer_name +"_bias.txt"))

	print("====================================")
	print("%s weights saved !!! " %netname)
	
	if WaitMode is True:	input("Please press the Enter key to proceed")
	
	
