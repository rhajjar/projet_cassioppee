# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:00:53 2019

@author: gourgue

code qui réuni les différentes fonction d'affichage pour les réseaux de neurones
plot train : affiche les courbes d'apprentissage et de validation de l'accuracy et loss
    au cours des différentes époques.
visu erreur : affiche les 5 première images mal classé dans les deux cas. faux positif
    et faux negatif avec la possibilité de les sauvegarder.
plot_matrix : affiche les matrices des performances avec le texte de nombre d'occurance 
    pour chaque case.
plot time : affiche la durée entre deux période
"""

#%%
#importation
from tensorflow.keras.models       import Model
from tensorflow.keras.callbacks    import History



import os ,sys, time
folder_code="D:/Users/gourgue/Documents/Nicolas/codes"
date=time.localtime()
date_str=str(date[0])+'-'+'%02d'%date[1]+'-'+'%02d'%date[2]
sys.path.append(os.path.join(folder_code,date_str+' compteur'))

import numpy as np

#from my_classes import DataGenerator, get_labels

import matplotlib.pyplot as plt

from skimage.io import imread, imsave

from my_cascade import get_labels_empty,  get_labels_infected

import datetime

#%%
def plot_train(model, title='train'):
    """
    prend en entrée un modèle entrainé ou l'historique et trace en fonction de ça.
    les courbes d'accuracy et d'erreur. 
    
    amélioration possible ne pas fixé en dur les métrics comme c'est fait mais 
    récupérer les clés du dictionnaire history.
    """
    if type(model)==Model:
        plt.figure(figsize=(30,10))
        plt.subplot(1,2,1)
        plt.plot(model.history.history["accuracy"])
        plt.plot(model.history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train","validation"],loc='lower right')
        plt.show()
        
        plt.subplot(1,2,2)
        plt.plot(model.history.history['loss'])
        plt.plot(model.history.history['val_loss'])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(['train','validation'], loc='upper left')
        plt.show()
    elif type(model)==History:
        plt.figure(figsize=(20,30))
        plt.subplot(1,2,1)
        plt.plot(model.history["acc"])
        plt.plot(model.history["val_acc"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train","validation"],loc='upper left')
        plt.show()
        
        plt.subplot(1,2,2)
        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(['train','validation'], loc='upper left')
        plt.show()
#%%
def VisuErreur(model, partition, testing_generator,folder_save='', mode='empty',SAVE=True,
               color='G'):
    #visualisation error
    """
    visu erreur permet de visualiser les erreurs faite par le réseaux. on tire les 5 premiers
    cas des deux type d'erreur possible. faux positif et faux negatif. ce code ne fonctionne
    que pour le classifieur en cascade
    model : réseau de neurone a tester.
    testing generator : generateur avec les données de teste. 
    folder_save : dossier de sauvegarde si données à sauvegarder
    mode : pour la cascade si on est dans le premier ou deuxième classifier
    save : si on sauvegarde les images ou pas.
    color : quel type de color son les images. 
    
    amélioration : change le mode pour généralisation la fonction et pourovir l'utiliser
    avec autre chose que le classifieur en cascade.
    """
    Y_pred = model.predict_generator(generator=testing_generator)
    Y_pred = np.argmax(Y_pred, axis=1)
    if mode=='empty':
        Y_test = get_labels_empty(partition['test'])[:len(Y_pred)]
    elif mode=='infected':
         Y_test = get_labels_infected(partition['test'])[:len(Y_pred)]
        
         
    Y_test = Y_test.reshape([len(Y_test),])
    Y_test = np.array(Y_test, dtype='uint64')

    f_neg = np.where(Y_test > Y_pred)
    f_pos = np.where(Y_test < Y_pred)
    part=np.array(partition['test'])
    image_neg_name = part[f_neg]
    image_pos_name = part[f_pos]
    
    image_neg = []
    image_pos = []
    for i in range(len(image_neg_name)):
        image_neg.append(imread(image_neg_name[i]))
    for i in range(len(image_pos_name)):    
        image_pos.append(imread(image_pos_name[i]))
     
        
    plt.figure()
    for j in range(min(5,len(image_neg))):
        plt.subplot(2,5,j+1)
        plt.imshow(image_neg[j], cmap='gray')
        plt.title('faux négatif')
        
    for j in range(5,min(10,5+len(image_pos))):
        plt.subplot(2,5,j+1)
        plt.imshow(image_pos[j-5], cmap='gray')
        plt.title("faux positif")
    plt.show()
    
    if SAVE:
        travel_output=folder_save+'image/'
        if not os.path.exists(travel_output):
            os.mkdir(travel_output)
        travel_output=travel_output+model.name+'/'
        if not os.path.exists(travel_output):
            os.mkdir(travel_output)
        #faux négatif
        travel_output_neg=travel_output+'/faux_neg/'
        if not os.path.exists(travel_output_neg):
            os.mkdir(travel_output_neg)
        for i,image in enumerate(image_neg):
            traveling, name_image = os.path.split(image_neg_name[i])
            traveling, folder = os.path.split(traveling)
            traveling, stat = os.path.split(traveling)
            traveling, champ = os.path.split(traveling)
            traveling, colori = os.path.split(traveling)
            traveling, patient = os.path.split(traveling)
            title=patient+'/'
            if not os.path.exists(travel_output_neg+title):
                os.mkdir(travel_output_neg+title)
            title=title+colori+'/'
            if not os.path.exists(travel_output_neg+title):
                os.mkdir(travel_output_neg+title)
            title=title+champ+'/'
            if not os.path.exists(travel_output_neg+title):
                os.mkdir(travel_output_neg+title)
            title=title+folder+'/'
            if not os.path.exists(travel_output_neg+title):
                os.mkdir(travel_output_neg+title)
            title=title+name_image
            imsave(travel_output_neg+title, image)
            
        #faux positif
        travel_output_pos=travel_output+'/faux_pos/'
        if not os.path.exists(travel_output_pos):
            os.mkdir(travel_output_pos)
        for i,image in enumerate(image_pos):
            traveling, name_image = os.path.split(image_pos_name[i])
            traveling, folder = os.path.split(traveling)
            traveling, stat = os.path.split(traveling)
            traveling, champ = os.path.split(traveling)
            traveling, colori = os.path.split(traveling)
            traveling, patient = os.path.split(traveling)
            title=patient+'/'
            if not os.path.exists(travel_output_pos+title):
                os.mkdir(travel_output_pos+title)
            title=title+color+'/'
            if not os.path.exists(travel_output_pos+title):
                os.mkdir(travel_output_pos+title)
            title=title+colori+'/'
            if not os.path.exists(travel_output_pos+title):
                os.mkdir(travel_output_pos+title)    
            title=title+color+'/'
            if not os.path.exists(travel_output_pos+title):
                os.mkdir(travel_output_pos+title)
            title=title+name_image
            imsave(travel_output_pos+title, image)        
        
#%%
def plot_matrix(matrix,title="matrice confusion", text=True):
    plt.figure()
    plt.imshow(matrix)
    plt.title(title)
    if text:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(i,j,str(matrix[j,i]))
    plt.colorbar()
    plt.axis('off')
    plt.xlabel("prediction")
    plt.ylabel("reality")
    plt.show()
#%%
    
def plot_time(debut, fin, title='duration of process'):
    print(title,datetime.timedelta(0,fin-debut))
    