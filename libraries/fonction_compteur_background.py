# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:21:26 2019

@author: gourgue
"""
#%%
from skimage.morphology import disk
from skimage.filters    import rank
import numpy as np
from fonction_compteur_affiche      import affi_image
#%%
def filtre_morpho(image, verbose=True, cells_mean=60):
    """ filtre_morpho : image with a threshold for the background.
    return image with background is 0.
    image      : image input
    verbose    : display process
    cells_mean : diameter of cells.
    """
    filter1 = rank.otsu(image, disk(2*cells_mean))
    sortie_seuiller=image<filter1
    result1=image*sortie_seuiller

    return result1

#%%
def try_filter (fonction, image, size, nb_time):
    """function for test morphological filter. and plot result
    no return
    fonction : function morphological
    image    : input image
    size     : is the size of disk for selem in function morphological
    nb_time  : number of times to applicate function morphological
    """
    color='gray'
    #calcul affichage
    rows=int(np.sqrt(nb_time+1))
    cols=int((nb_time+1)/rows)+1
    
    for j in size:
        titre=titre='image with '+str(fonction)[9:str(fonction).find('at')]+' disk='
        titre=titre+str(j)
        fig18=affi_image(image, title=titre, cols=cols, rows=rows, close=False)
        for i in range(nb_time):
            image_filtrer=fonction(image,disk(j ))
            title='image with disk='
            title=title+str(j)
            title=title+' '
            title=title+str(i+1)
            title=title+' times'
            axes8=fig18.add_subplot(rows,cols,i+2, label='filter1 ostu')
            axes8.set_title(title)
            axes8.imshow(image_filtrer, cmap=color)
            axes8.axis("off")
            
#%%
def filtre_back(image, verbose=True, cells_mean=60):
    filter1 = rank.otsu(image, disk(2*cells_mean))
    return filter1