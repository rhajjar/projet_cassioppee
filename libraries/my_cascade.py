# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:41:11 2019

@author: gourgue
"""

#%%

from tensorflow.keras.utils import Sequence, to_categorical

from skimage.io import imread

import numpy as np

import os

class DataGenerator_empty(Sequence):
    
    def __init__(self, list_IDs, batch_size=64, dim=[84,84], n_channels=3, n_classes=2, 
                 shuffle=True, empty='empty', LEDS=False):
        'Initialisation'
        self.batch_size = batch_size
        self.dim        = dim
        self.list_IDs   = list_IDs
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.shuffle    = shuffle
        self.LEDS       = LEDS
        self.on_epoch_end()
        self.empty=empty
        
        
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)
            
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        #Initialisation
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        
        #Generate data
        #after led
        X=get_images(list_IDs_temp, self.n_channels, self.LEDS)
        if self.empty=='empty':
            y=get_labels_empty(list_IDs_temp)
        elif self.empty=='infected':
            y=get_labels_infected(list_IDs_temp)
        elif self.empty=='3':
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
    
   

def get_labels_empty(liste):
    y=np.zeros([len(liste),1])
    for i, path in enumerate(liste):
        travel, name =os.path.split(path)
        if 'healthy' in name:
            y[i]=0
        elif 'infected' in name:
            y[i]=1
        elif 'distrac' in name:
            y[i]=1
    return y

def get_images(liste, nb_channels, LED=False):
    green=False
    color=False
    ptycho=False
    gris_3D=False
    for i, path in enumerate(liste):
        image = imread(path)
        if i==0:
            #premier tour initialisation
            if nb_channels == 1:
                #cas image niveau de gris
                if len(image.shape)==2:
                    #image 2D niveau de gris
                    X=np.zeros([len(liste), image.shape[0], image.shape[1]])
                elif image.shape[2]==1:
                    gris_3D = True
                    #image 3D niveau de gris
                    X=np.zeros([len(liste), image.shape[0], image.shape[1]])
                elif image.shape[2]==3:
                    #image 3D convertion en niveau de gris. par défaut on prendra le canal vert
                    green = True
                    X=np.zeros([len(liste), image.shape[0], image.shape[1]])
                else:
                    #problème
                    print("problème : l'image lu n'est pas en niveau de gris")
                    
            elif nb_channels == 3:
                #cas d'une image couleur
                if len(image.shape)==2:
                    #cas d'une image 2D convertion en couleur
                    color = True
                    X=np.zeros([len(liste), image.shape[0], image.shape[1],3])
                elif image.shape[2]==1:
                    #cas d'une image 2D convertion en couleur
                    gris_3D = True
                    color   = True
                    X=np.zeros([len(liste), image.shape[0], image.shape[1],3])
                elif image.shape[2]==3:
                    #cas d'une image 3D
                    X=np.zeros([len(liste), image.shape[0], image.shape[1],3])
                else:
                    print("problème : l'image lu n'est pas en niveau couleur ni en niveau de gris")
            
            elif nb_channels > 3:
                ptycho=True
                #cas des images en ptycographie
                if len(image.shape)==2:
                    #cas d'une image 2D 
                    X=np.zeros([len(liste), image.shape[0], image.shape[1], nb_channels])
                elif image.shape[2]==1:
                    #cas d'une image 2D
                    gris_3D = True
                    X=np.zeros([len(liste), image.shape[0], image.shape[1], nb_channels])
                elif image.shape[2]==3:
                    #cas d'une image 3D convertion en 2D
                    green = True
                    X=np.zeros([len(liste), image.shape[0], image.shape[1], nb_channels])
                else:
                    print("problème : l'image lu n'est pas en niveau couleur ni en niveau de gris")  
                
            else:
                print("problème votre image n'as pas les dimension requis",image.shape)
        
        if green and ptycho is False:
            X[i,:,:]=image[:,:,1]
        elif color and ptycho is False:
            if gris_3D:
                X[i,:,:,0] = image[:,:,0]
                X[i,:,:,1] = image[:,:,0]
                X[i,:,:,2] = image[:,:,0]
            else:
                X[i,:,:,0] = image
                X[i,:,:,1] = image
                X[i,:,:,2] = image
        elif gris_3D and ptycho is False:
            X[i,:,:,0]=image[:,:,0]
            X[i,:,:,1]=image[:,:,0]
            X[i,:,:,2]=image[:,:,0]
            
        elif ptycho:
            #cas ptycho
            path_picture , image_name = os.path.split(path)
            if 'infected' in image_name:    
                path_stat , folder_image  = os.path.split(path_picture)
                path_field , folder_stat  = os.path.split(path_stat)
                path_color , folder_field = os.path.split(path_field)
#                X_temp=np.zeros([image.shape[0], image.shape[1], nb_channels])
                for j, diode in enumerate (LED):
                    image_bis = imread(path_color+'/'+'RAW_'+str(diode)+'/'+folder_stat+'/'+\
                                           folder_image+'/'+image_name)
                    if gris_3D:
                        X[i,:,:,j]=image_bis[:,:,0]
                    elif green:
                        X[i,:,:,j]=image_bis[:,:,1]
                    else:
                        X[i,:,:,j]=image_bis
            
            elif 'healthy' in image_name:
                 path_field , folder_image  = os.path.split(path_picture)
                 path_color , folder_field = os.path.split(path_field)
#                 X_temp=np.zeros([image.shape[0], image.shape[1], nb_channels])
                 for j, diode in enumerate (LED):
                    image_bis = imread(path_color+'/'+'RAW_'+str(diode)+'/'+\
                                           folder_image+'/'+image_name)
                    if gris_3D:
                        X[i,:,:,j]=image_bis[:,:,0]
                    elif green:
                        X[i,:,:,j]=image_bis[:,:,1]
                    else:
                        X[i,:,:,j]=image_bis
        else:
            X[i,]=image
            
            
                    
    return X/255


def get_labels(liste):
    y=np.zeros([len(liste),1])
    for i, path in enumerate(liste):
        travel, name =os.path.split(path)
        if 'healthy' in name:
            y[i]=0
        elif 'infected' in name:
            y[i]=2
        elif 'distrac' in name:
            y[i]=1
    return y

def get_labels_infected(liste):
    y=np.zeros([len(liste),1])
    for i, path in enumerate(liste):
        travel, name =os.path.split(path)
        if 'healthy' in name:
            y[i]=0
        elif 'infected' in name:
            y[i]=1
        elif 'distrac' in name:
            y[i]=0
    return y