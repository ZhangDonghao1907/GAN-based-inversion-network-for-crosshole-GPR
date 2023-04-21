# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 16:24:35 2023

@author: Zhang Donghao
"""
import keras
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras import layers
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def define_discriminator(image_shape):
	init = RandomNormal(stddev=0.02)
	in_src_image = Input(shape=(64,64,36))
	in_target_image = Input(shape=(64,64,16))
	merged = Concatenate()([in_src_image, in_target_image])
	fc1 =keras.layers.Dense(256 , activation='relu')(merged)
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(fc1)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('relu')(d)
	model = Model([in_src_image, in_target_image], patch_out)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='mean_squared_error', optimizer=opt, loss_weights=[100])
	model.summary()
	return model

def define_rdiscriminator(image_shape):
	init = RandomNormal(stddev=0.02)
	in_src_image = Input(shape=(256,256,1))
	in_target_image = Input(shape=(256,256,1))
	merged = Concatenate()([in_src_image, in_target_image])
	fc1 =keras.layers.Dense(256 , activation='relu')(merged)
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(fc1)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('relu')(d)
	model = Model([in_src_image, in_target_image], patch_out)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='mean_squared_error', optimizer=opt, loss_weights=[100])
	model.summary()
	return model

def define_encoder_block(layer_in, n_filters, batchnorm=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	g = LeakyReLU(alpha=0.2)(g)
	return g

def define_encoder_block_1(layer_in, n_filters, batchnorm=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	g = LeakyReLU(alpha=0.2)(g)
	return g

def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Concatenate()([g, skip_in])
	g = Activation('relu')(g)
	return g


def decoder_block_none(layer_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Activation('relu')(g)
	return g

def decoder_block_none_1(layer_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Activation('relu')(g)
	return g


def decoder_block_1(layer_in, skip_in, n_filters, dropout=True):
	init = RandomNormal(stddev=0.02)
	g = Conv2DTranspose(n_filters, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(layer_in)
	g = BatchNormalization()(g, training=True)
	if dropout:
		g = Dropout(0.5)(g, training=True)
	g = Concatenate()([g, skip_in])
	g = Activation('relu')(g)
	return g


# define the standalone generator model
def define_generator(image_shape=(64,64,36)):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=(36,4096))
	fc_1= layers.Reshape((4096, 36))(in_image) 
	fc_1 =keras.layers.Dense(1024, activation='relu')(fc_1)
	fc_4 = layers.Reshape((64, 64, 1024))(fc_1)
	model_Res_conv = keras.applications.resnet_v2.ResNet101V2(input_shape=(64,64,1024), include_top=False, weights=None, input_tensor=None)
	output_RES51_conv = model_Res_conv(fc_4)
	d1 = layers.Reshape((2, 2, 2048))(output_RES51_conv)
	d1 = decoder_block_none_1(d1, 256)
	d1_1 = decoder_block_none_1(d1, 256)
	d2 = decoder_block_none(d1_1, 128)
	d2_1 = decoder_block_none_1(d2, 128)
	d3 = decoder_block_none(d2_1, 64)#64
	d3_1 = decoder_block_none_1(d3, 64)#64
	d4 = decoder_block_none(d3_1, 64)#64
	d4_1 = decoder_block_none_1(d4, 64)#64
	d5 = decoder_block_none(d4_1, 64)#64
	d5_1 = decoder_block_none_1(d5, 64)#64
	d6 = decoder_block_none(d5_1, 32)
	g = Conv2DTranspose(1, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(d6)
	fea = Activation('relu')(g)
	y1 = define_encoder_block(fea, 16)#32
	y2 = define_encoder_block(y1, 32)#16
	y3 = define_encoder_block(y2, 64)#8
	y4 = define_encoder_block(y3, 128)#4
	y5 = define_encoder_block_1(y4, 256)#4
	c1 = Concatenate()([fea, d6])#64
	c2 = Concatenate()([y1, d5])#32
	c3 = Concatenate()([y2, d4])#16
	c4 = Concatenate()([y3, d3])#8
	c5 = Concatenate()([y4, d2])#4
	de1 = decoder_block_none_1(y5, 256)#4
	con2 = Concatenate()([c5, de1])#4
	de2 = decoder_block_none(con2, 128)#4
	con3 = Concatenate()([c4, de2])#8
	de3 = decoder_block_none(con3, 64)#16
	con4 = Concatenate()([c3, de3])
	de4 = decoder_block_none(con4, 32)#32
	con5 = Concatenate()([c2, de4])
	de5 = decoder_block_none(con5, 32)#64
	con6 = Concatenate()([c1, de5])
	de6 = decoder_block_none(con6, 16)#128
	de7 = decoder_block_none(de6, 32)#256
	g1 = Conv2DTranspose(1, (1,1), strides=(1,1), padding='same', kernel_initializer=init,activation='relu')(de7)
	model = Model(in_image,g1,name="modellow")
	model.summary()
	return model


def define_refiner(image_shape):
	init = RandomNormal(stddev=0.02)
	in_image = Input(shape=(256,256,1))
	in_image1 = Concatenate()([in_image, in_image, in_image])
	base_model = keras.applications.resnet_v2.ResNet50V2(input_shape=(256,256,3), include_top=False, weights='imagenet', input_tensor=in_image1)
	base_model.summary()
	x1 = base_model.get_layer("conv1_conv").output
	x2 = base_model.get_layer("pool1_pool").output
	x3 = base_model.get_layer("conv2_block3_out").output
	x4 = base_model.get_layer("conv3_block4_out").output
	x5 = base_model.get_layer("conv4_block6_out").output
	x6 = base_model.get_layer("conv5_block3_out").output
	d1_1 = decoder_block_none_1(x6, 256)#2
	concat2 = Concatenate()([x5, d1_1])#2
	d2 = decoder_block_none(concat2, 128)#4
	d2_1 = decoder_block_none_1(d2, 128)#4
	concat3 = Concatenate()([x4, d2_1])
	d3 = decoder_block_none(concat3, 64)#8
	d3_1 = decoder_block_none_1(d3, 64)#8
	concat4 = Concatenate()([x3, d3_1])
	d4 = decoder_block_none(concat4, 32)#16
	d4_1 = decoder_block_none_1(d4, 32)#16
	concat5 = Concatenate()([x2, d4_1])
	d5 = decoder_block_none(concat5, 16)#32
	d5_1 = decoder_block_none_1(d5, 16)#32
	concat6 = Concatenate()([x1, d5_1])
	d6 = decoder_block_none(concat6, 8)#32
	g1 = Conv2DTranspose(1, (1,1), strides=(1,1), padding='same', kernel_initializer=init)(d6)
	fea = Activation('relu')(g1)
	y1 = define_encoder_block(fea, 16)#32
	y2 = define_encoder_block(y1, 32)#16
	y3 = define_encoder_block(y2, 64)#8
	y4 = define_encoder_block(y3, 128)#4
	y5 = define_encoder_block_1(y4, 256)#4
	c1 = Concatenate()([fea, d6])#64
	c2 = Concatenate()([y1, d5])#32
	c3 = Concatenate()([y2, d4])#16
	c4 = Concatenate()([y3, d3])#8
	c5 = Concatenate()([y4, d2])#4
	de1 = decoder_block_none_1(c5, 256)#4
	con2 = Concatenate()([y5, de1])#2
	de2 = decoder_block_none_1(con2, 128)#4
	con3 = Concatenate()([c5, de2])
	de3 = decoder_block_none(con3, 64)#8
	con4 = Concatenate()([c4, de3])
	de4 = decoder_block_none(con4, 32)#16
	con5 = Concatenate()([c3, de4])
	de5 = decoder_block_none(con5, 16)#32
	con6 = Concatenate()([c2, de5])
	de6 = decoder_block_none(con6, 8)#64
	con7 = Concatenate()([c1, de6])
	g2 = Conv2DTranspose(1, (1,1), strides=(1,1), padding='same', kernel_initializer=init, activation='sigmoid')(con7)
	model = Model(in_image,g2)
	model.summary()
	return model

def define_gan(g_model, d_model1,image_shape):
	for layer in d_model1.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	in_src1 = Input(shape=image_shape)
	gen_out = g_model(in_src1)
	in_src = layers.Reshape((64,64, 36))(in_src1)
	gen_out1 = layers.Reshape((64,64,16))(gen_out)
	dis_out = d_model1([in_src, gen_out1])
	model = Model(in_src1, [dis_out,gen_out])
	opt =keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=['MAE','MSE'], optimizer=opt, loss_weights=[0,100])
	return model

def define_ref(r_model, d_model2,image_shape):
	for layer in d_model2.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	in_src = Input(shape=image_shape)
	ref_out = r_model(in_src)
	dis_out = d_model2([in_src, ref_out])
	model = Model(in_src, [dis_out,ref_out])
	opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	model.compile(loss=['MAE','MSE'], optimizer=opt, loss_weights=[0,100])
	return model

def load_real_samples(filename):
	data = load(filename)
	X1,X2 = data['arr_0'], data['arr_1']
	return [X1, X2]

def generate_real_samples(dataset, n_samples, patch_shape):
	trainA, trainB = dataset
	ix = randint(0, trainA.shape[0], n_samples)
	X1, X2= trainA[ix], trainB[ix]
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

def generate_fake_samples(g_model, samples, patch_shape):
	X = g_model.predict(samples)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def generate_refine_samples(r_model, samples, patch_shape):
	X = r_model.predict(samples)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def summarize_performance(step, g_model,r_model,dataset, n_samples=3):
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	X_fakeBs, _ = generate_refine_samples(r_model, X_fakeB, 1)
	filename2 = 'g%06d.h5' % (step+1)
	filename1 = 'r%06d.h5' % (step+1)
	g_model.save(filename2)
	r_model.save(filename1)
	print('>Saved: %s' %  filename2)

def train(d_model1,d_model2, g_model,gan_model,r_model,rgan_model,dataset, n_epochs=500, n_batch=3):
	n_patch = d_model1.output_shape[1]
	n_patch1 = d_model2.output_shape[1]
	trainA, trainB = dataset
	bat_per_epo = int(len(trainA) / n_batch)
	n_steps = bat_per_epo * n_epochs
	for i in range(n_steps):
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		X_fakeBs, y_fakes = generate_refine_samples(r_model, X_fakeB, n_patch1)
		X_realAr =  layers.Reshape((64,64, 36))(X_realA)
		X_realBr =  layers.Reshape((64,64, 16))(X_realB)
		X_fakeBr =  layers.Reshape((64,64, 16))(X_fakeB)
		d_loss1 = d_model1.train_on_batch([X_realAr, X_realBr], y_real)
		d_loss2 = d_model1.train_on_batch([X_realAr, X_fakeBr], y_fake)
		d_loss3 = d_model2.train_on_batch([X_fakeB, X_realB], y_real)
		d_loss4 = d_model2.train_on_batch([X_fakeB, X_fakeBs], y_fakes)
		print('%d %.3f %.3f %.3f %.3f' % (i+1, d_loss1, d_loss2, d_loss3, d_loss4))
		if (i+1) % (bat_per_epo * 100) == 0:
			summarize_performance(i, g_model,r_model, dataset)

dataset = load_real_samples('yourdata.npz')
image_shape = dataset[0].shape[1:]
print(image_shape)
d_model1 = define_discriminator(image_shape)
d_model2 = define_rdiscriminator(image_shape)
g_model = define_generator(image_shape)
r_model = define_refiner((256,256,1))
gan_model = define_gan(g_model,d_model1,image_shape)
rgan_model = define_ref(r_model,d_model2,(256,256,1))
train(d_model1, d_model2,g_model, gan_model, r_model,rgan_model,dataset)
