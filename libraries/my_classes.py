# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:49:49 2019

@author: gourgue
adapter le code pour les images compressers.
"""
#%%

from keras.utils import Sequence, to_categorical

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img,save_img

from skimage.io import imread

import numpy as np

import os, time, zlib

import cv2

from osgeo import gdal

import matplotlib.pyplot as plt

from PIL import Image

class DataGenerator(Sequence):
    
    def __init__(self, list_IDs, batch_size=64, dim=[84,84], n_channels=3, n_classes=2, 
                 shuffle=True, LEDS=False):
        'Initialisation'
        self.batch_size = batch_size
        self.dim        = dim
        self.list_IDs   = list_IDs
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.LEDS       = LEDS
        self.on_epoch_end()
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.seed(1)
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        #Initialisation
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        #Generate data
        #after led
        X=get_images(list_IDs_temp, self.n_channels, self.LEDS)
        y=get_labels(list_IDs_temp)
            
        return X, to_categorical(y, num_classes=self.n_classes)
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__ (self, index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        #Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
   

def get_labels(liste):
    y=np.zeros([len(liste),1])
    for i, path in enumerate(liste):
        travel, name =os.path.split(path)
        if 'healthy' in name:
            y[i]=0
        elif 'infected' in name:
            y[i]=1
    return y

def get_images(liste, nb_channels, LED=False):
    
    for i, path in enumerate(liste):
        #image=imread(path,pilmode="RGB",as_gray=True)
        image = np.array(Image.open(path))
        
        if i==0:
            if LED is False:
              if(image.shape[0])==35:
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
              else:
                X=np.zeros([len(liste), image.shape[0], image.shape[1],nb_channels])
            elif LED == "multi_led":
#                image=np.moveaxis(image,0,2)
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
            else:
#                image=np.moveaxis(image,0,2)
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
        
        if LED is False:
            if(image.shape[0])==35:
                X[i,:,:,0]=image[0,:,:]
                X[i,:,:,1]=image[0,:,:]
                X[i,:,:,2]=image[0,:,:]
            elif len(image.shape)==3:
                X[i]=image
            elif len(image.shape)==2:
                X[i,:,:,0]=image
                X[i,:,:,1]=image
                X[i,:,:,2]=image
        elif LED =='multi_led':
            image=np.moveaxis(image,0,2)
            X[i]=image[:,:,:nb_channels]
        else:
            image=np.moveaxis(image,0,2)
            X[i]=image[:,:,LED]          
                    
    return X/255#X/127-1

class DataGeneratorPhase(Sequence):
    
    def __init__(self, list_IDs, batch_size=64, dim=[84,84], n_channels=3, n_classes=2, 
                 shuffle=True, LEDS=False):
        'Initialisation'
        self.batch_size = batch_size
        self.dim        = dim
        self.list_IDs   = list_IDs
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.LEDS       = LEDS
        self.on_epoch_end()
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.seed(1)
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        #Initialisation
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        #Generate data
        #after led
        X=get_imagesPhase(list_IDs_temp, self.n_channels, self.LEDS)
        y=get_labels(list_IDs_temp)
            
        return X, to_categorical(y, num_classes=self.n_classes)
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__ (self, index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        #Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

def get_imagesPhase(liste, nb_channels, LED=False):
    
    for i, path in enumerate(liste):
        path_phase = path.replace("inten","phase")
        
        image= np.array(Image.open(path))
        image_phase = np.array(Image.open(path_phase))

        nule = np.zeros([image.shape[0], image.shape[1]])
        if i==0:
            if LED is False:
              if(image.shape[0])==35:
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
              else:
                X=np.zeros([len(liste), image.shape[0], image.shape[1],nb_channels])
            elif LED == "multi_led":
#                image=np.moveaxis(image,0,2)
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
            else:
#                image=np.moveaxis(image,0,2)
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
        
        if LED is False:
            if(image.shape[0])==35:
                X[i,:,:,0]=image[0,:,:]
                X[i,:,:,1]=image[0,:,:]
                X[i,:,:,2]=image[0,:,:]
            elif len(image.shape)==3:
                X[i]=image
            elif len(image.shape)==2:
                X[i,:,:,0]=image
                X[i,:,:,1]=image_phase
                X[i,:,:,2]=nule
        elif LED =='multi_led':
            image=np.moveaxis(image,0,2)
            X[i]=image[:,:,:nb_channels]
        else:
            image=np.moveaxis(image,0,2)
            X[i]=image[:,:,LED]          
                    
    return X/255#X/127-1

class DataGeneratorTopHat(Sequence):
    
    def __init__(self, list_IDs, batch_size=64, dim=[84,84], n_channels=3, n_classes=2, 
                 shuffle=True, LEDS=False):
        'Initialisation'
        self.batch_size = batch_size
        self.dim        = dim
        self.list_IDs   = list_IDs
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.LEDS       = LEDS
        self.on_epoch_end()
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.seed(1)
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        #Initialisation
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        #Generate data
        #after led
        X=get_images_tophat(list_IDs_temp, self.n_channels, self.LEDS)
        y=get_labels(list_IDs_temp)
            
        return X, to_categorical(y, num_classes=self.n_classes)
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__ (self, index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        #Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y
    
    
def get_images_tophat(liste, nb_channels, LED=False):

    for i, path in enumerate(liste):
        image = imread(path)
        if i==0:
            #premier tour initialisation
            X = np.zeros([len(liste), image.shape[0], image.shape[1],2])
        X[i,:,:,0]=image
        traveling, name = os.path.split(path)
        traveling+='_tophat/'
        tophat = imread(traveling+name)
        X[i,:,:,1]=tophat
        
        
        
    return X/255    


class DataGeneratorCentral(Sequence):
    
    def __init__(self, list_IDs, batch_size=64, dim=[84,84], n_channels=3, n_classes=2, 
                 shuffle=True, LEDS=False):
        'Initialisation'
        self.batch_size = batch_size
        self.dim        = dim
        self.list_IDs   = list_IDs
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.LEDS       = LEDS
        self.on_epoch_end()
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.seed(1)
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        #Initialisation
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        #Generate data
        #after led
        X=get_imagesCentral(list_IDs_temp, self.n_channels, self.LEDS)
        y=get_labels(list_IDs_temp)
            
        return X, to_categorical(y, num_classes=self.n_classes)
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__ (self, index):
        'Generate one batch of data'
        #Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        #Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        #Generate data
        X, y = self.__data_generation(list_IDs_temp)
        
        return X, y

def get_imagesCentral(liste, nb_channels, LED=False):
    
    for i, path in enumerate(liste):
        image=imread(path)
        #image = np.array(Image.open(path))
        
        if i==0:
            if LED is False:
              if(image.shape[0])==35:
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
              else:
                X=np.zeros([len(liste), image.shape[0], image.shape[1],nb_channels])
            elif LED == "multi_led":
#                image=np.moveaxis(image,0,2)
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
            else:
#                image=np.moveaxis(image,0,2)
                X=np.zeros([len(liste), image.shape[1], image.shape[2],nb_channels])
        
        if LED is False:
            if(image.shape[0])==35:
                X[i,:,:,0]=image[0,:,:]
                X[i,:,:,1]=image[0,:,:]
                X[i,:,:,2]=image[0,:,:]
            elif len(image.shape)==3:
                X[i]=image
            elif len(image.shape)==2:
                X[i,:,:,0]=image
                X[i,:,:,1]=image
                X[i,:,:,2]=image
        elif LED =='multi_led':
            image=np.moveaxis(image,0,2)
            X[i]=image[:,:,:nb_channels]
        else:
            image=np.moveaxis(image,0,2)
            X[i]=image[:,:,LED]          
                    
    return X/255#X/127-1