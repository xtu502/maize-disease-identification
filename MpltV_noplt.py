# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:23:24 2019

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 07:11:11 2018

@author: HP
"""

import tensorflow as tf
tf.reset_default_graph()
#from attention_module import *
import keras as kr

from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
import keras.backend.tensorflow_backend as KTF
import glob
from tensorflow.keras.layers import Input,Dense,Dropout,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D,concatenate,Activation,ZeroPadding2D
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Dense, Lambda

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from skimage import io,data
import time
#from keras import layers
#from keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow import keras 
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier



import os,sys
os.getcwd()
os.chdir("/home/cjd/33_Col_Maize")
print(os.getcwd())
print (sys.version)


now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
#import os
# 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
kr.backend.tensorflow_backend.set_session(tf.Session(config=config))



import tensorflow as tf        
def focal_loss(gamma=2.):            
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        return -K.sum( K.pow(1. - pt_1, gamma) * K.log(pt_1)) 
    return focal_loss_fixed


def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):  
    if name is not None:  
        bn_name = name + '_bn'  
        conv_name = name + '_conv'  
    else:  
        bn_name = None  
        conv_name = None  
  
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)  
    x = BatchNormalization(axis=3,name=bn_name)(x)  
    return x  

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):  
    x = Conv2d_BN(inpt,nb_filter=nb_filter[0],kernel_size=(1,1),strides=strides,padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3,3), padding='same')  
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1,1), padding='same')  
    if with_conv_shortcut:  
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter[2],strides=strides,kernel_size=kernel_size)  
        x = add([x,shortcut])  
        return x  
    else:  
        x = add([x,inpt])  
        return x  


#---------------------------------Attention mechanism-----------------------------------------------------------
import tensorflow.contrib.slim as slim

def cbam(inputs):
    inputs_channels=int(inputs.shape[-1])
    x=keras.layers.GlobalAveragePooling2D()(inputs)
    x=keras.layers.Dense(int(inputs_channels/4))(x)
    x=keras.layers.Activation('relu')(x)
    x=keras.layers.Dense(int(inputs_channels))(x)
    x=keras.layers.Activation('softmax')(x)
    x=keras.layers.Reshape((1,1,inputs_channels))(x)
    x=keras.layers.Multiply()([inputs,x])
    return x


class SeBlock(keras.layers.Layer):   
    def __init__(self, reduction=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction = reduction
    def build(self,input_shape):
    	#input_shape     
    	pass
    def call(self, inputs):
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(int(x.shape[-1]) // self.reduction, use_bias=False,activation=keras.activations.relu)(x)
        x = keras.layers.Dense(int(inputs.shape[-1]), use_bias=False,activation=keras.activations.hard_sigmoid)(x)
        return keras.layers.Multiply()([inputs,x])    
        #return inputs*x 


batch_size = 64 
epochs = 30
MODEL_INIT = './obj_reco/init_model.h5'
MODEL_PATH = './obj_reco/tst_model.h5'
board_name1 = './obj_reco/stage1/' + now + '/'
board_name2 = './obj_reco/stage2/' + now + '/'

train_dir1='/home/wkq/Projects/kerasVGG19/train_b/'
validation_dir1='/home/wkq/Projects/kerasVGG19/test_b/'

train_dir2='/home/cjd/33_Col_Maize/train_seg/'
validation_dir2='/home/cjd/33_Col_Maize/test_seg/'

img_size = (224, 224)  
#classes=list(range(1,5))
#classes=['1','2','3','4']

nb_train_samples1 = len(glob.glob(train_dir1 + '/*/*.*'))  
nb_validation_samples1 = len(glob.glob(validation_dir1 + '/*/*.*'))  
classes1 = sorted([o for o in os.listdir(train_dir1)])  

nb_train_samples2 = len(glob.glob(train_dir2 + '/*/*.*'))  
nb_validation_samples2 = len(glob.glob(validation_dir2 + '/*/*.*'))  
classes2 = sorted([o for o in os.listdir(train_dir2)])  



#-----------#Mobile-DANet--------------------------------------------------------------
def get_bottlenet(image_size,alpha=1.0):
    inputs = keras.layers.Input(shape=(image_size,image_size,3),name='input_1')
    net = keras.layers.ZeroPadding2D(padding=(3,3),name='zero_padding2d_1')(inputs)
    net = keras.layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2),
                             padding='valid', name='conv1/conv')(net)
    net = keras.layers.BatchNormalization(name='conv1/bn')(net)
    net = keras.layers.ReLU(name='conv1/relu')(net)
    net = keras.layers.ZeroPadding2D(padding=(1,1),name='zero_padding2d_2')(net)

 #-----------------attention block----------------------------------------------------------------------------
    residual = tf.keras.layers.Conv2D(64, kernel_size = (1, 1), strides = (1, 1), padding = 'same')(net)
    residual = tf.keras.layers.BatchNormalization(axis = -1)(residual)
    cbam_mdl = cbam(residual)
#    cbam = spatial_attention_module(residual, kernel_size=7, reuse=None, scope='spatial_attention')
    Atten = tf.keras.layers.add([net, residual, cbam_mdl])
#---------------------------------------------------------------------------------------------------------------
    Cam = keras.layers.GlobalAveragePooling2D(name='transform2_pool')(Atten)

    net = keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='valid',
                                   name='pool1')(net)

    for i in range(int(3*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32,kernel_size=(3,3), strides=(1, 1), 
                                    padding='same',
                                    name='conv2_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv2_block{}_concat'.format(i))([net,block])    
    net = keras.layers.BatchNormalization(name='pool2_bn')(net) 
    net = keras.layers.ReLU(name='pool2_relu')(net)
    cam_f = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform2_dense1')(Cam)
    net = keras.layers.Multiply(name='transform2_multiply')([net,cam_f])


    net = keras.layers.Conv2D(filters=int(net.shape[-1])//2,kernel_size=(1,1),strides=(1,1),
                              padding='same',name='pool2_conv')(net)
    net = keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),
                                        name='pool2_pool')(net)

#-----------------attention block----------------------------------------------------------------------------
    Atten = cbam(net)
    Cam = keras.layers.GlobalAveragePooling2D(name='transform3_pool')(Atten)
#---------------------------------------------------------------------------------------------------------------

    for i in range(int(6*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), strides=(1, 1),
                                             padding='same',
                                             name='conv3_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv3_block{}_concat'.format(i))([net,block])
    net = keras.layers.BatchNormalization(name='pool3_bn')(net) 
    net = keras.layers.ReLU(name='pool3_relu')(net)
    cam_f = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform3_dense1')(Cam)
    net = keras.layers.Multiply(name='transform3_multiply')([net,cam_f])

    net = keras.layers.Conv2D(filters=int(net.shape[-1])//2,kernel_size=(1,1),strides=(1,1),
                              padding='same',name='pool3_conv')(net)
    net = keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),
                                    name='pool3_pool')(net)

#-----------------attention block----------------------------------------------------------------------------
    Atten = cbam(net)
    Cam = keras.layers.GlobalAveragePooling2D(name='transform4_pool')(Atten)
#---------------------------------------------------------------------------------------------------------------

    for i in range(int(12*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
                                    padding='same',
                                    name='conv4_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv4_block{}_concat'.format(i))([net,block])
    net = keras.layers.BatchNormalization(name='pool4_bn')(net) 
    net = keras.layers.ReLU(name='pool4_relu')(net)
    cam_f  = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform4_dense1')(Cam)
    net = keras.layers.Multiply(name='transform4_multiply')([net,cam_f])

    net = keras.layers.Conv2D(filters=int(net.shape[-1])//2,kernel_size=(1,1),strides=(1,1),
                             padding='same',name='pool4_conv')(net)

    net = keras.layers.AveragePooling2D(pool_size=(2,2),strides=(2,2),
                                        name='pool4_pool')(net)
#-----------------attention block----------------------------------------------------------------------------
    Atten = cbam(net)
    Cam = keras.layers.GlobalAveragePooling2D(name='transform5_pool')(Atten)
#---------------------------------------------------------------------------------------------------------------

    for i in range(int(8*alpha)):
        block = net
        block = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), strides=(1, 1), 
                                    padding='same',
                                    name='conv5_block{}_2_sepconv'.format(i))(block)
        net = keras.layers.Concatenate(name='conv5_block{}_concat'.format(i))([net,block])
       
    cam_f = keras.layers.Dense(units=net.shape[-1],activation='sigmoid',
                            name='transform5_dense1')(Cam)
    net = keras.layers.Multiply(name='transform5_multiply')([net,cam_f])
    net = keras.layers.BatchNormalization(name='bn')(net)
    net = keras.layers.ReLU(name='relu')(net)
    model = keras.Model(inputs=inputs,outputs=net,name='mobile_densenet_bottle')
    return model

def get_model1(image_size=224,alpha=1.0,classes=10):
    bottlenet = get_bottlenet(alpha=alpha,image_size=image_size)
    net = keras.layers.GlobalAveragePooling2D(name='global_pool')(bottlenet.output)
    net = keras.layers.Dropout(rate=0.4,name='dropout1')(net)
    net = keras.layers.Dropout(rate=0.4,name='dropout2')(net)
    output = keras.layers.Dense(units=classes,activation='softmax',
                             name='prediction1')(net)
    model = keras.Model(inputs=bottlenet.input,outputs=output,name='mobile_densenet')
    return model


def get_model2(image_size=224,alpha=1.0,classes=10):
    bottlenet = get_bottlenet(alpha=alpha,image_size=image_size)
    net = keras.layers.GlobalAveragePooling2D(name='global_pool')(bottlenet.output)
    net = keras.layers.Dropout(rate=0.4,name='dropout1')(net)
    net = keras.layers.Dropout(rate=0.4,name='dropout2')(net)
    output = keras.layers.Dense(units=classes,activation='softmax',
                             name='prediction2')(net)
    model = keras.Model(inputs=bottlenet.input,outputs=output,name='mobile_densenet')
    return model


model1 = get_model1(image_size=224,classes=len(classes1))
model2 = get_model2(image_size=224,classes=len(classes2))
#model2.get_layer(name='fc1').name='mlp_0_2'



train_datagen = ImageDataGenerator(validation_split=0.2)
train_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))  
train_data1 = train_datagen.flow_from_directory(train_dir1, target_size=img_size, classes=classes1)
train_data2 = train_datagen.flow_from_directory(train_dir2, target_size=img_size, classes=classes2)

validation_datagen = ImageDataGenerator()
validation_datagen.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))
validation_data1 = validation_datagen.flow_from_directory(validation_dir1, target_size=img_size, classes=classes1)
validation_data2 = validation_datagen.flow_from_directory(validation_dir2, target_size=img_size, classes=classes2)

'''
# -------------1st stage-----------------
#model_checkpoint1 = ModelCheckpoint(filepath=MODEL_INIT, save_best_only=True, monitor='val_accuracy', mode='max')
model_checkpoint1 = ModelCheckpoint(filepath=MODEL_INIT, monitor='val_accuracy')
board1 = TensorBoard(log_dir=board_name1,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
callback_list1 = [model_checkpoint1, board1]

#model1.compile(optimizer=keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics=['accuracy'])
model1.compile(optimizer='adam', loss=[focal_loss(gamma=2)],  metrics = ['accuracy'])

model1.fit_generator(train_data1, steps_per_epoch=nb_train_samples1 / float(batch_size),
                           epochs = epochs,
                           validation_steps=nb_validation_samples1 / float(batch_size),
                           validation_data=validation_data1,
                           callbacks=callback_list1, verbose=2)

for lys in model1.layers:
     lys._name = lys._name+str('_1')
'''
#---------------2nd stage---------------------------------------------
model_checkpoint2 = ModelCheckpoint(filepath=MODEL_PATH,  monitor='val_accuracy')
board2 = TensorBoard(log_dir=board_name2,
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)
callback_list2 = [model_checkpoint2, board2]


model2.load_weights(MODEL_INIT,by_name=True)
for lyrs in model2.layers:
#    lyrs.name = lyrs.name + str('_2')
    lyrs.trainable = True

'''
#-----tensorflow keras -----------------------
#learning_rate = 0.01
learning_rate = 0.0001
decay = 1e-6
momentum = 0.9
nesterov = True
sgd_optimizer = keras.optimizers.SGD(lr = learning_rate, decay = decay,            
                    momentum = momentum, nesterov = nesterov)
model2.compile(metrics = ['accuracy'],                           
                               loss = [focal_loss(gamma=2)],
#                               loss = 'categorical_crossentropy',
                               optimizer = sgd_optimizer)
'''


model2.compile(optimizer='adam', loss=[focal_loss(gamma=2)],  metrics = ['accuracy'])  #rmsprop
#model2.compile(optimizer=keras.optimizers.Adam(), loss =[focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy', Adadelta
#model2.compile(optimizer=optimizers.SGD(lr=0.0001), loss = [focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',
##model.compile(optimizer=optimizers.Adadelta(), loss = [focal_loss(gamma=2)], metrics=['accuracy']) #loss='categorical_crossentropy',


model2.fit_generator(train_data2,steps_per_epoch=nb_train_samples2/float(batch_size),epochs=epochs,
                    validation_data=validation_data2,validation_steps=nb_validation_samples2/float(batch_size),
                    callbacks=callback_list2,                  
                    verbose=2)


'''
from contextlib import redirect_stdout   
with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model2.summary(line_length=200,positions=[0.30,0.60,0.7,1.0])

from tensorflow.keras.utils import  plot_model
#from keras.utils import plot_model
plot_model(model2,to_file="model.png",show_shapes=True)
'''


