# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:09:21 2022

@author: pc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:51:16 2021

@author: pc
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 15:35:56 2021

@author: pc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 19:37:52 2021

@author: pc
"""

# example of pix2pix gan for satellite to map image-to-image translation
import tensorflow as tf
import keras

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras import layers
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Conv2DTranspose
from keras.layers import Conv3DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from matplotlib import pyplot
from keras.utils.vis_utils import plot_model
# load, split and scale the maps dataset ready for training
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import skimage.io as io
import numpy as np
#config = ConfigProto()
#config.gpu_options.allow_growth = False
#session = InteractiveSession(config=config)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# dataset path
from keras.models import load_model
#def load_images(path, size=(256, 512)):
	#src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	#for filename in listdir(path):
		# load and resize the image
		#pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		#pixels = img_to_array(pixels)
		# split into satellite and map
		#sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		#src_list.append(sat_img)
		#tar_list.append(map_img)
	#return [asarray(src_list), asarray(tar_list)]

# define the discriminator model

def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=(64,64,36))
	
	# target image input
	in_target_image = Input(shape=(64,64,64))
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	fc1 =keras.layers.Dense(256 , activation='relu')(merged)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(fc1)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('relu')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='msle', optimizer=opt, loss_weights=[100])
	model.summary()
	return model
'''
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=(64,64,36))
	
	# target image input
	in_target_image = Input(shape=(64,64,64))
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	merged =keras.layers.Reshape((64,64,100,1))(merged)
	fc1 =keras.layers.Dense(16 * 16 , activation='relu')(merged)
	fc1 =keras.layers.Reshape((64,64,64,400))(fc1)
	d1 = Conv3D(16, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(fc1)
	d1 = LeakyReLU(alpha=0.2)(d1)
	# C128
	d2 = Conv3D(32, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d1)
	d2 = BatchNormalization()(d2)
	d2 = LeakyReLU(alpha=0.2)(d2)
    
	# C256
	d3 = Conv3D(64, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d2)
	d3 = BatchNormalization()(d3)
	d3 = LeakyReLU(alpha=0.2)(d3)
	# C512
	d1res = Conv3D(64, (1,1,1), strides=(4,4,4), padding='same', kernel_initializer=init, activation='relu')(d1)
	d1add = Add()([d3,d1res])
	
    
	d4 = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d1add)
	d4 = BatchNormalization()(d4)
	d4 = LeakyReLU(alpha=0.2)(d4)
	# second last output layer
	d5 = Conv3D(128, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(d4)
	d5 = BatchNormalization()(d5)
	d5 = LeakyReLU(alpha=0.2)(d5)
	d2res = Conv3D(128, (1,1,1), strides=(4,4,4), padding='same', kernel_initializer=init, activation='relu')(d4)
	d2add = Add()([d5,d2res])
	# patch output
	d6 = Conv3D(1, (4,4,4), padding='same', kernel_initializer=init)(d2add)
	patch_out = Activation('relu')(d6)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(lr=0.01, beta_1=0.5)
	model.compile(loss='msle', optimizer=opt, loss_weights=[1000])
	model.summary()
	return model
'''
# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

def define_encoder_block_1(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g



def define_refiner_Conv3d(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv3D(n_filters, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

def define_refiner_Conv3d_1(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv3D(n_filters, (1,1,1), strides=(1,1,1), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g
# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
    #g = Concatenate()([g, skip_in])
    #g = tf.concat([g, skip_in], axis = 3)

	# relu activation
	g = Activation('relu')(g)
	return g



def decoder_block_none(layer_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	#g = Concatenate()([g, skip_in])
    #g = Concatenate()([g, skip_in])
    #g = tf.concat([g, skip_in], axis = 3)

	# relu activation
	g = Activation('relu')(g)
	return g


def decoder_block_none_1(layer_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	#g = Concatenate()([g, skip_in])
    #g = Concatenate()([g, skip_in])
    #g = tf.concat([g, skip_in], axis = 3)

	# relu activation
	g = Activation('relu')(g)
	return g


def decoder_block_1(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
    #g = Concatenate()([g, skip_in])
    #g = tf.concat([g, skip_in], axis = 3)

	# relu activation
	g = Activation('relu')(g)
	return g

def td_decoder_block(layer_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv3DTranspose(n_filters, (4,4,4), strides=(2,2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	#g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

def td_decoder_block_1(layer_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv3DTranspose(n_filters, (1,1,1), strides=(1,1,1), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	#g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g
def convolution_block(
    block_input,
    num_filters=512,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)
def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output
def DeeplabV3Plus(image_size, num_classes):
    model_input = keras.Input(shape=(image_size, image_size, 30))
    
	# encoder model
		 
    in_image1 = layers.Reshape((64,64,30))(model_input)
    resnet50 = keras.applications.ResNet50(
        weights=None, include_top=False, input_tensor=in_image1
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    return keras.Model(inputs=model_input, outputs=model_output)



def define_pix2vox(image_shape=(128,128,3)):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=image_shape)
	# encoder model
	
	fc_1 = layers.Reshape((49152, ))(in_image)
	
	
	#fc_1 = keras.Input(shape=(24576))
	#with tf.device('/gpu:0'):
	fc_2 = layers.Dense(128 * 128 * 3, activation='tanh')(fc_1)
	
	
	
	fc_3 = layers.Dense(128 * 128 * 3, activation='tanh')(fc_2)
	
	fc_3 = layers.Reshape((128, 128, 3))(fc_3)
	
	e1 = define_encoder_block_1(fc_3, 64)#128
	
	e1_1 = define_encoder_block(e1, 128)#64
	e1_2 = define_encoder_block_1(e1_1, 128)
	
	e2_1 = define_encoder_block(e1_2, 128)#32
	
	
	e2_2 = define_encoder_block_1(e2_1, 256)
	
	e3_1 = define_encoder_block_1(e2_2, 256)
	
	e3_2 = define_encoder_block(e3_1, 256)#16
	#c3 = define_encoder_block_1(e3_2, 4)
	e4_1 = define_encoder_block_1(e3_2, 256)
	
	e4_2 = define_encoder_block_1(e4_1, 256)
	
	e5_1 = define_encoder_block_1(e4_2, 256)
	e5_2 = define_encoder_block_1(e5_1, 256)
	
	e6_1 = define_encoder_block_1(e5_2, 256)
	e6_2 = define_encoder_block_1(e6_1, 256)
	
	e7_1 = define_encoder_block_1(e6_2, 256)
	e7_2 = define_encoder_block_1(e7_1, 256)
	
	e8_1 = define_encoder_block_1(e7_2, 256)
	e8_2 = define_encoder_block_1(e8_1, 256)
	
	e9_1 = define_encoder_block_1(e8_2, 256)
	e9_2 = define_encoder_block_1(e9_1, 256)
	
	e10_1 = define_encoder_block_1(e9_2, 128)
	e10_2 = define_encoder_block_1(e10_1, 64)
	
	#concat1 = Concatenate()([c3, e10_2])
	
	volume1 = layers.Reshape((2, 2, 2, 2048))(e10_2)
	#decoder
	d1 = td_decoder_block(volume1, 512)#4
	d2 = td_decoder_block(d1, 128)#8
	d3 = td_decoder_block(d2, 32)#16
	d4 = td_decoder_block(d3, 8)#32
	d5 = Conv3DTranspose(1, (1,1), strides=(1,1), padding='same', kernel_initializer=init, activation='sigmoid')(d4)
	raw_features = Concatenate()([d4, d5])
	#merge
	m1 = define_refiner_Conv3d(raw_features, 16)
	m2 = define_refiner_Conv3d(m1, 8)
	m3 = define_refiner_Conv3d(m2, 4)
	m4 = define_refiner_Conv3d(m3, 2)
	m5 = define_refiner_Conv3d(m4, 1)
	#Refiner
	
	rc1 = define_refiner_Conv3d(m5, 32)
	rc2 = define_refiner_Conv3d(rc1, 64)
	rc3 = define_refiner_Conv3d(rc2, 128)
	
	rfc1 = layers.Dense(8192, activation='relu')(rc3)
	rfc2 = layers.Dense(8192, activation='relu')(rfc1)
	rfc2 = layers.Reshape((4, 4, 4, 128))(rfc2)
	rd1 = td_decoder_block(rfc2, 64)
	rd2 = td_decoder_block(rd1, 32)
	rd3 = Conv3DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init, activation='sigmoid')(rd2)
	
	# define model
	model = Model(in_image, rd3)
	model.summary()
	
	return model






# define the standalone generator model
def define_generator(image_shape=(64,64,3)):
	
    # weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=(36,4096))
	# encoder model
	fc_1= layers.Reshape((4096, 36))(in_image) 
	#fc_1 = layers.Reshape((4096,1))(in_image)
	#in_image11 = layers.Reshape((196608, ))(in_image1)
	#in_image21 = layers.Reshape((196608, ))(in_image2)
	#in_image31 = layers.Reshape((196608, ))(in_image3)
	#pool2 = keras.layers.TimeDistributed(keras.layers.MaxPooling2D(pool_size=(2, 2)))(e2)#16
	fc_1 =keras.layers.Dense(1024, activation='relu')(fc_1)
	#fc_2 = layers.Reshape((64, 64, 3))(fc_2)
	#fc_2 = layers.Reshape((12288, ))(fc_2)
	#fc_3 = keras.layers.TimeDistributed(keras.layers.Dense(4096, activation='relu'))(l1)
	#fc_4 = layers.Dense(300, activation='relu')(merged_vector)
	fc_4 = layers.Reshape((64, 64, 1024))(fc_1)
	model_Res_conv = keras.applications.resnet_v2.ResNet101V2(input_shape=(64,64,1024), include_top=False, weights=None, input_tensor=fc_4)
	model_Res_conv.summary()
	x = model_Res_conv.get_layer("post_relu").output
	#output_RES51_conv = model_Res_conv(fc_4)

	#fc1 = layers.Reshape((16640, ))(concat1)
	#fc2 = layers.Dense(8 * 8 * 260, activation='relu')(fc1)
	#fc2 = layers.Reshape((8, 8, 260))(fc2)
	#fc2 = layers.Reshape((16640, ))(fc2)
	#fc3 = layers.Dense(8 * 8 * 260, activation='relu')(fc2)
	#fc3 = layers.Reshape((8, 8, 260))(fc3)
	output_RES51_conv = layers.Reshape((4, 4, 4, 128))(x)
	de11 = td_decoder_block(output_RES51_conv, 128)#8
	de12 = td_decoder_block(de11, 128)#16
	de13= td_decoder_block_1(de12, 128)#16
	de1res = Conv3DTranspose(128, (1,1,1), strides=(2,2,2), padding='same', kernel_initializer=init, activation='relu')(de11)
	d1add = Add()([de13,de1res])
	de1 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid')(d1add)
	#8
	de21 = td_decoder_block(de1, 64)#8
	de22 = td_decoder_block(de21, 64)#4
	de23 = td_decoder_block_1(de22, 64)
	de2res = Conv3DTranspose(64, (1,1,1), strides=(2,2,2), padding='same', kernel_initializer=init, activation='relu')(de21)
	d2add = Add()([de23,de2res])
	de2 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid')(d2add)
    #16              
	de31 = td_decoder_block(de2, 32)#16
	de32 = td_decoder_block(de31, 32)#4
	de33 = td_decoder_block_1(de32, 32)#4
	de3res = Conv3DTranspose(32, (1,1,1), strides=(2,2,2), padding='same', kernel_initializer=init, activation='relu')(de31)
	d3add = Add()([de33,de3res])
	de3 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid')(d3add)
	#32
	de41 = td_decoder_block(de3, 16)#32
	de42 = td_decoder_block(de41, 16)#4
	de43 = td_decoder_block_1(de42, 16)#4
	de4res = Conv3DTranspose(16, (1,1,1), strides=(2,2,2), padding='same', kernel_initializer=init, activation='relu')(de41)
	d4add = Add()([de43,de4res])
	de4 = keras.layers.MaxPooling3D(pool_size=(2,2,2), strides=None, padding='valid')(d4add)
	#64
	de5 = td_decoder_block_1(de4, 8)#32
	de5 = Conv3DTranspose(1, (1,1,1), strides=(1,1,1), padding='same', kernel_initializer=init, activation='sigmoid')(de5)
	
    
	#raw_features = Concatenate()([de5, de51])
	
	#fc_2 =keras.layers.Dense(1 , activation='sigmoid')(de5)
	#fc_2 =keras.layers.Dense(16, activation='relu')(de4)
	#fc_3 =keras.layers.Dense(8, activation='relu')(fc_2)
	#fc_44 =keras.layers.Dense(1, activation='sigmoid')(fc_3)
	'''
	#merge
	m1 = define_refiner_Conv3d_1(raw_features, 16)#64
	m2 = define_refiner_Conv3d_1(m1, 8)
	m3 = define_refiner_Conv3d_1(m2, 4)
	m4 = define_refiner_Conv3d_1(m3, 2)
	m5 = define_refiner_Conv3d_1(m4, 1)
	out_image = Activation('sigmoid')(m5)
	'''
    
	model = Model(in_image,de5,name="model3d")
	model.summary()
   
    
    
	return model


def define_refiner(image_shape):
	init = RandomNormal(stddev=0.02)
	
	in_image = Input(shape=(64,64,3))
	rc1 = define_encoder_block(in_image, 64)#32
	rc1_1 = define_encoder_block_1(rc1, 64)#32
	#rc1 = keras.layers.MaxPooling2D(pool_size=(2), strides=None, padding='valid')(rc1)
	rc2 = define_encoder_block(rc1_1, 128)#16
	rc2_1 = define_encoder_block_1(rc2, 128)#16
	#rc2 = keras.layers.MaxPooling2D(pool_size=(2), strides=None, padding='valid')(rc2)
	rc3 = define_encoder_block(rc2_1, 256)#8
	rc3_1 = define_encoder_block_1(rc3, 256)#8
	#rc3 = keras.layers.MaxPooling2D(pool_size=(2), strides=None, padding='valid')(rc3)
	
	
	rd1 = decoder_block_none(rc3_1, 256)#16
	rhc1 = Concatenate()([rd1, rc2_1])
	rd2 = decoder_block_none(rhc1, 128)#32
	rhc2 = Concatenate()([rd2, rc1_1])
	rd3 = decoder_block_none_1(rhc2, 64)#64
	rd4 = Conv2DTranspose(64, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(rd3)
	
	# define model
	model = Model(in_image, rd4)
	model.summary()
	#opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,)
	#model.compile(loss=['MSE'], optimizer=opt, loss_weights=[10])
	return model






# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model,image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src1 = Input(shape=image_shape)
	
	# connect the source image to the generator input
	gen_out = g_model(in_src1)
	# connect the source input and generator output to the discriminator input
	#in_src = Concatenate()([in_src1, in_src2,in_src3])
	in_src = layers.Reshape((64,64, 36))(in_src1)
	gen_out1 = layers.Reshape((64,64,64))(gen_out)
	dis_out = d_model([in_src, gen_out1])
	
	#ref_out = r_model(gen_out)
	# src image as input, generated image and classification output
	model = Model(in_src1, [dis_out,gen_out])
	# compile model
	opt =keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=['MAE','MSLE'], optimizer=opt, loss_weights=[0,1000])
	return model

def define_ref(gan_model, r_model, image_shape):
	# make weights in the discriminator not trainable
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = gan_model(in_src)[1]
	# connect the source input and generator output to the discriminator input
	ref_out = r_model(gen_out)
	
	#ref_out = r_model(gen_out)
	# src image as input, generated image and classification output
	model = Model(in_src, ref_out)
	# compile model
	opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=['MAE', 'MSE'], optimizer=opt, loss_weights=[0,10])
	return model



# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	#X1 = (X1 - 127.5) / 127.5
	#X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y




# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def generate_refine_samples(ref_model, samples, patch_shape):
	# generate fake instance
	X = ref_model.predict(samples)
	# create 'fake' class labels (0)
	#y = zeros((len(X), patch_shape, patch_shape, 1))
	return X



# generate samples and save as a plot and save the model
def summarize_performance(step, g_model,dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	#X_fakeBR = generate_refine_samples(ref_model, X_realA, 1)
	'''
	PSNR=0
	SSIM=0
	MSE =0
	for i in range(n_samples):
		
		psnr_val = peak_signal_noise_ratio(X_fakeB[i], X_realB[i])
		PSNR = PSNR+psnr_val
		ssim_val = structural_similarity(X_fakeB[i], X_realB[i],win_size=11,gaussian_weights=True,multichannel=True,data_range=1.0,K1=0.01,K2=0.03,sigma=1.5)
		SSIM = SSIM+ssim_val
		mse_val = np.mean( (X_fakeB[i]- X_realB[i]) ** 2 )
		MSE = MSE + mse_val
	print("psnr_val",PSNR/3)
	
	print("ssim_val",SSIM/3)
	
	print("MSE",MSE/3)
	with open("eva_new.txt","a+") as f:
		f.write('%.3f %.3f %.3f \n'%(PSNR/3,SSIM/3,MSE/3)
               )
	
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB[X_fakeB<-1] = -1
	X_fakeB = (X_fakeB + 1) / 2.0
	#X_fakeBR[X_fakeBR<-1] = -1
	#X_fakeBR = (X_fakeBR + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	
	# plot generated target image
	#for i in range(n_samples):
		#pyplot.subplot(3, n_samples, 1 + i)
		#pyplot.axis('off')
		#pyplot.imshow(X_fakeBR[i])
	
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	'''
	# save plot to file
	#filename1 = '3%06d.png' % (step+1)
	#pyplot.savefig(filename1)
	#pyplot.close()
	# save the generator model
	#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
	filename2 = 'true3d%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s' %  filename2)

# train pix2pix models
def train(d_model, g_model,gan_model,dataset, n_epochs=500, n_batch=3):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		#X_fakeBR, y_fake = generate_refine_samples(r_model, X_fakeB, n_patch)
		# update discriminator for real samples
		X_realAr =  layers.Reshape((64,64, 36))(X_realA)
		X_realBr =  layers.Reshape((64,64, 64))(X_realB)

		d_loss1 = d_model.train_on_batch([X_realAr, X_realBr], y_real)
		X_fakeBr =  layers.Reshape((64,64, 64))(X_fakeB)
		d_loss2 = d_model.train_on_batch([X_realAr, X_fakeBr], y_fake)
		#d_loss3 = d_model.train_on_batch([X_realA, X_fakeBR], y_fake)
		# update the generator
		#g_loss = g_model.train_on_batch(X_realA,X_realB)
		gan_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		#ref_loss= ref_model.train_on_batch(X_realA, X_realB)
		#g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB,X_fakeB])
		#g_loss1, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] gan[%.3f]' % (i+1, d_loss1, d_loss2,gan_loss))
		with open("testbg1.txt","a+") as f:
			f.write('%.3f %.3f %.3f\n'%(d_loss1, d_loss2,gan_loss)
                    )
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)

# load image data
dataset = load_real_samples('maps_lstmgtrue3d.npz')

print('Loaded', dataset[0].shape, dataset[1].shape)
# define input shape based on the loaded dataset
image_shape = dataset[0].shape[1:]
print(image_shape)
# define the models
d_model = define_discriminator((16,16,4))
g_model = define_generator(image_shape)
#r_model = define_refiner(g_model)

# define the composite model
gan_model = define_gan(g_model,d_model,image_shape)
#ref_model = define_ref(gan_model, r_model, image_shape)
# train model
train(d_model, g_model, gan_model, dataset)