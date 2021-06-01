# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:47:24 2019

@author: gourgue

première cellule : importation
deuxième cellule : paramètres
    path : chemin de l'image
    travel_labeled : dossier de sauvegarde pour la carte de segmentation
    travel_para : dossier de sauvegarde du csv contenant les coordonnées des
                  parasites
    travel_dataset : dossier de sauvegarde des imagettes après segmentation
troisème cellule : lecture de l'image original et sauvegarde en grand format
quatrième cellule : réglage des différents paramètres pour la segmentation
                    ouverture de l'image et exécutation de la segmenatation
cinquième cellule : tentative de traiter l'image par morceau. non fini
"""
#%% importation 
import os 
from skimage.io import imread, imsave
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from fonction_compteur import process, ouvrir

#%% paramètres
path="D:/Users/gourgue/Documents/Nicolas/alexander code/comparasion/in.tif"
travel_labeled="D:/Users/gourgue/Documents/Nicolas/alexander code/comparasion/"
travel_dataset="D:/Users/gourgue/Documents/Nicolas/alexander code/comparasion/"
travel_para=False

travels=[path,travel_labeled,travel_dataset, travel_para]
#%%

image_0 = imread(path)
if len(image_0.shape)==3:
    image_0=image_0[0,:,:]

image_1 = Image.fromarray(image_0)

image_2 = image_1.resize([image_1.width*2, image_1.height*2])#, Image.BILINEAR)

image_3 = np.array(image_2)


plt.figure()
plt.imshow(image_3, cmap='gray')

traveling, name=os.path.split(path)
imsave(traveling+"/"+name+"_resize.png",image_3)
#%%

images=[]
para=False 
segmentation='calcul' 
travels=[]

travel_input=traveling+"/"+name+"_resize.png"
travel_labeled=traveling+"/"
travel_dataset=traveling+"/"
travel_para=False
image_name=travel_input
travels=[travel_input,travel_labeled,travel_dataset,travel_para]
size =10/1.2
Save=True
raw=0
verbose=True
temps=True
exemple=[False]

image, titles, traveling=ouvrir(travel_input, name="manual")
plt.figure()
plt.imshow(image, cmap='gray')


process(image_name, para=para,segmentation=segmentation,size=size, Save=Save, verbose=verbose, 
                        temps=temps, travels=travels , exemple=exemple, titles=titles)

#%%
for i in range(6):
    for j in range(6): 
        image_name=image[int(image.shape[0]/6*i):int(image.shape[0]/6*(i+1)),int(image.shape[1]/6*j):\
                             int(image.shape[1]/6*(j+1))]
        image_name=np.array(image_name)
#        plt.figure()
#        plt.imshow(image_name, cmap='gray')
        process(image_name, para=para,segmentation=segmentation,size=size, Save=Save, verbose=verbose, 
                        temps=temps, travels=travels , exemple=exemple, titles=titles)