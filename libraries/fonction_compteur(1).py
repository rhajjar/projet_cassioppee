# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:06:50 2019

@author: gourgue
cellule 1 : importation
"""
#%%
#importation

import numpy as np

import os

from skimage.io         import imread, imsave
from skimage.color      import rgb2gray
from skimage.measure    import regionprops
from skimage            import img_as_uint

import matplotlib.pyplot as plt

from fonction_compteur_segmentation import Hough_by_thres, recon_image, test_region, detect_para
from fonction_compteur_datagenerator import decoupe_mask, sauvegarde_imagette, extract_para, extract_imagette
from fonction_compteur_affiche import affiche,draw_ellipse_perso

import time

import pandas as pd
#fonction open for extract file
#%%
def ouvrir(image,name="automatic"):
    """ function for open image. 
    return image to matrix, title image and travel image.
    image : travel
    name  : automatic for add last folder with the title of picture or
    manual for extract in title only title.
    """
    #if we input a travel on image
    if type(image)==str:
        in_dir, image_name = os.path.split(image)
        if name=="automatic":
            travel, folder_name=os.path.split(in_dir)
            title, ext=os.path.splitext(image_name)
            title=folder_name+' '+title
        elif name=='manual':
            travel=in_dir
            title, ext=os.path.splitext(image_name)
        else:
            travel=None
            print("erreur, name = 'automatic' or 'manual'. here name=",name)
            return None
            
        image= imread(image)
        
        #cas des images multidimensionnel
        if len(image.shape)==3:
            if image.shape[0]==3 or image.shape[-1]==3:
                #image de couleur
                if image.shape[-1]==3:
                    #convertion en nuance de gris
                    image= rgb2gray(image)
                else :
                    #trouver comment convertir quand la couleur est en premier
                    pass
            elif image.shape[0]>image.shape[-1]:
                #multi image moyenne sur la dernière composante
#                image=np.mean(image, axis=2)
                image=image[:,:,:]
            elif image.shape[0]<image.shape[-1]:
#                image=np.mean(image, axis=0)
                image=image[:,:,:]
            else:
                pass
            
                

        return image, title, travel
    elif type(image)==np.ndarray:
        #if image in color convert to gray image
        if len(image.shape)==3:
            image= rgb2gray(image)
            return image, 'image',None
        elif len(image.shape)==2:
            return image, 'image', None
        else :
            print("you are a problem of size image")
            print("size image :",image.shape)
            return None
    else :
        print("l'image n'est pas un array classique ni un chemin type=",type(image))
        print("le traitement n'est pas sur de réussir")
        return image, 'image', None

#%%
def sauver(travel_output, title, image, classe):
    """ function for save a picture wih ellipse in each cells detected. 
    return image
    travel_output : travel where image is save
    title         : title of image
    image         : image save with ellipse for each cells
    classe        : list of region as detected as cells.
    """
    if not os.path.exists(travel_output):
            os.mkdir(travel_output)
    if type(image)==type(plt.figure()):
        image.savefig(travel_output+title+'.png')
    elif type(image)==type(np.array([0,0])):
        image=image*255
        image=draw_ellipse_perso(image, classe)
        imsave(travel_output+title+'.png', image)
        print(travel_output)
    return image


    
#%%
def complet_cells(classe, labeled, verbose=True):
    """ function for conserve only the convexe envelop for cells.
    return matrix with convex envelop for label
    classe  : list of region as cells
    labeled :matrix with label
    verbose : display process
    """
    
    labeled_after=labeled.copy()
    for region in classe:
        boxe=region.bbox
        labels_alone=region.label
        labeled_rogner=labeled_after[boxe[0]:boxe[2],boxe[1]:boxe[3]]
        labeled_rogner[region.convex_image]=labels_alone
        
        if verbose=='all':
            plt.figure("labeled_rogner")
            plt.imshow(labeled_rogner)
            
            plt.figure("labeled rogner après traitement")
            plt.imshow(labeled_rogner)
    if verbose==True or verbose=='all':
        plt.figure("labeled before")
        plt.imshow(labeled)
        
        plt.figure("labeled_after")
        plt.imshow(labeled_after)
    return labeled_after

#%%
def process(image_name, para='csv',segmentation='extract',size=10/1.2, Save=False, verbose=False, temps=True, 
            travels=os.getcwd(), exemple=[False], raw=0, titles=None, tophat=False):
    """
    cette fonction est le processus de segmentation pour une seule image.
    image_name : le nom ou l'image en entrée.
    para       : c'est la manière dont on annote les parasites. 
        csv : signifie qu'il exsite un fichier csv qui regroupe les coordonnées pour l'image des parasites.
        tophat : c'est le début de la fonction base sur un tophat sombre pour détecté les parastes de manière
                 automatique. non abouti
        False  : pas de parasites référencé sur l'image
    segmentation : manière de découpé la grande image.
        calcul   : on utilise différente méthode morphologique et transformation de hough pour extraire
                   les cellules.
        extract  : on utilise une image de référence ou les cellules sont déjà découpées. 
    size     : resolution de l'image en pixel par micron.
    Save     : en cas de calcul de la segmentation est ce qu'on sauvegarde la carte de résultat
    verbose  : affichage
        False: on affiche la représentation en couleur de la carte de segmentation sur fond d'image d'origine.
        all  : on affiche toute les étapes de la segmentation (utilisé uniquement pour le débugage)
        True : on affiche les étapes importante de la segmentation
    temps : on mesure le temps et on l'affiche à la fin.
    travels : on envoie les différents chemin utile pour la suite : chemin d'entrée de l'image, chemin
        pour récupérer ou sauvegarder la carte, chemin pour savegarder le dataset, chemin pour prendre
        le fichier csv des parasites.
    exemple : arguments a supprimé
    raw  : diode a visualisé 
    titles : titre a mettre sur les illustrations
    tophat : calcul et sauvegarde de l'image tophat
    
    """
    #begining 
    if temps:
        debut=time.time()
        
    cells_mean=size*7.2
        
    if type(image_name)==str:
        
        #extract image
        image, titles, traveling=ouvrir(image_name, name="manual")
        if image.shape[0]==35:
            image=image[raw,:,:]
    
    elif type(image_name)==np.ndarray:
        image=image_name

    #normalise image
    if image.dtype!='uint8':
        img_cr=(image -image.min())/(image.max()-image.min())
        image=img_cr*255
        del(img_cr)
        image=np.array(image, dtype='uint8')
        
    
    
    if verbose:
        plt.figure()
        plt.imshow(image, cmap='gray')
    if exemple[0]:
        imsave(exemple[1]+str(exemple[2])+'.png', image)
        exemple[2]+=1
    #    origine=image.copy()
    '''
    #file reference
    if segmentation=='extract':
        travel_extract=travels[0]
        travel_extract, folder=os.path.split(travel_extract)
        travel_extract, folder=os.path.split(travel_extract)
        traveling, file=os.path.split(travel_extract)
        table=pd.read_csv(travel_extract+"/"+file+".txt", sep='\t')
        
        columns=table.columns
        indice=table.where(titles==table[columns[0]])
        result=indice.dropna()
        champ=result[columns[2]]
        champ=str(champ.values[0])
        
        ref_champ=table.where(champ==table[columns[2]])
        ref_champ=ref_champ.dropna()
        ref_color=ref_champ.where("G"==ref_champ[columns[-1]])
        ref_color=ref_color.dropna()
        ref_moda =ref_color.where("FPM"==ref_color[columns[-2]])
        ref_moda =ref_moda.dropna()
        
        titles_extract=str(ref_moda[columns[0]].values[0])
    else:
        titles_extract=titles
        
    '''    
    #travels
    if type(travels)==list:
#        travel                = travels[0]
        travel_labeled = travels[1]
        travel_dataset = travels[2]+titles+"/" 
        travel_para    = travels[3]
        if travel_para is None :
            para=False
    else:
        travel_labeled = travels
        travel_dataset = travels+titles+"/" 
        travel_para    = travels
        
    #create au mask for background
    if para=='tophat':
        tophat =True
    if tophat or segmentation=='calcul':
        image_mask=decoupe_mask(image, verbose=verbose)
        zoro=np.ma.masked_where(image_mask==0,image_mask)

        if verbose:
            plt.figure()
            plt.imshow(zoro, cmap='gray')  
            origine_zoro=zoro.copy()
        if exemple[0]:
            imsave(exemple[1]+str(exemple[2])+'.png', zoro)
            exemple[2]+=1
        
    #extract parasite coords.
    if para=='csv':
    #    travel_para=(os.getcwd()+'/coords_para.csv')
        coords_para=extract_para(travel_para+titles_extract+'.csv')
        coords_para=np.array(coords_para, dtype='uint32')
    elif para=='tophat':
        zoro=origine_zoro
        verbose='all'
        coords_para=detect_para (image,zoro, verbose=verbose, title=titles, thres=6)#, champ=champ)
        coords_para=np.array(coords_para, dtype='uint32').T
    elif para is False:
        coords_para=False
    else :
        print("problem para=",para)
        
    #extract distrac coords.
    if "distrac" in travel_dataset:
        travel_distrac = travel_labeled[:-8]+"distrac csv/"
        coords_distrac = extract_para(travel_distrac+titles_extract+'.csv')
        coords_distrac = np.array(coords_distrac, dtype='uint32')
    else:
        coords_distrac=False
    #segmentation
    if segmentation=='calcul':
        labeled_list=Hough_by_thres(image,zoro,condition=['seuil',1,0.9],edges=None,labeled_list=[], 
                                    verbose=False, exemple=exemple)   
#        origine_labeled_list=labeled_list.copy() 
        labeled_list=recon_image(labeled_list,verbose=verbose)
        labeled=labeled_list[0] 
        #test if region is too small ot too big
        classe, amas_cells=test_region(labeled, cells_mean=cells_mean, threshold='hough_iter', bord=True)
        result=affiche(image, labeled,classe, title=titles+" premier filtrage",boxes=["ellipse","blue"])
        save=result[1]
        save=np.array(save*255, dtype='uint8')
        save=draw_ellipse_perso(save, classe)
        if not os.path.exists(travel_labeled):
            os.mkdir(travel_labeled)
        if Save:
            imsave(travel_labeled+titles+"_cells.png", save)
        save=draw_ellipse_perso(save, amas_cells)
        if Save:
            imsave(travel_labeled+titles+" cells and clusters.png", save)
        #conserve convexe area of segmentation.
        labeled_conv=complet_cells(classe+amas_cells, labeled, verbose=verbose)
        print('nb cells detected = ',len(classe))
        print('nb cluster detected = ',len(amas_cells))
        classe=regionprops(labeled_conv)
        labeled_conv = img_as_uint(labeled_conv)
        imsave(travel_labeled+titles+" labeling.png", labeled_conv)
        #save dataset segmentation
        _=sauvegarde_imagette(image,zoro, classe+amas_cells,coords_para=coords_para,
                              coords_distrac=coords_distrac,cells_mean=cells_mean, size1=84 ,size2=84,
                                     travel_output=travel_dataset)
        
    elif segmentation=='extract':
        labeled=imread(travel_labeled)
        classe=regionprops(labeled)
        print('nb cells detected = ',len(classe))
        if tophat:
            _=extract_imagette(image, labeled, coords_para, coords_distrac,cells_mean=cells_mean, size1=84, size2=84, 
                                 travel_output=travel_dataset, zoro=zoro, tophat=tophat)
        else:
            _=extract_imagette(image, labeled, coords_para, coords_distrac,cells_mean=cells_mean, size1=84, size2=84, 
                                 travel_output=travel_dataset)
        
         
    if temps:
        fin=time.time()
        print("traitement "+titles,fin-debut)
    return None


#%% 
def list_image_path(patient, color, champ, raw=0 ):
    """
    calcul des différents paramètre pour automatisé la fonction process
    patient : dossier patient a traité :"CAT01 KPJ0 DA LE"
    color   : dossier couleur a traité :"R G B"
    champ   : dossier champ a triaté   :"BF DF RAW"
    raw     : dans le cas du champ RAW numéro de la diode
    """
    travel="/content/drive/My Drive/Stage/Segmentation/Data/"
    travel_input="/content/drive/My Drive/Stage/BDD_malaria_original/"
    patients=['KPJ0','CAT01','DA','LE']
    colors=["R","G","B"]
    champs=["BF","DF","RAW"]
    
    if patient==patients[1]:
        travel_input=travel_input+"MAlDetect_CAT01-4/"
        travel_dataset=travel+"CAT01/"
        travel_para=travel+"CAT01/para csv/"
        travel_labeled=travel+"CAT01/labeled/"
        para='csv'
            
    elif patient==patients[0]:
        travel_input=travel_input+"MalDetect_KPJ0/"
        travel_dataset=travel+"KPJ0/"    
        travel_para=travel+"KPJ0/para csv/"
        travel_labeled=travel+"KPJ0/labeled/"
        para='csv'
        
    elif patient==patients[2]:
        travel_input=travel_input+"MalDetect_DA270519/"
        travel_dataset=travel+"DA/"
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
        travel_para=None
        travel_labeled=travel+"DA/labeled/"
        if not os.path.exists(travel_labeled):
            os.mkdir(travel_labeled)
        para=False
    
    elif patient==patients[3]:
        travel_input=travel_input+"MalDetect_LE300519/"
        travel_dataset=travel+"LE/"
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
        travel_para=None
        travel_labeled=travel+"LE/labeled/"
        if not os.path.exists(travel_labeled):
            os.mkdir(travel_labeled)
        para=False
        
    if color==colors[0]:
        travel_input=travel_input+'R/'
        travel_dataset=travel_dataset+'R/'
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
        
    elif color==colors[1]:
        travel_input=travel_input+'G/'
        travel_dataset=travel_dataset+'G/'
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
    elif color==colors[2]:
        travel_input=travel_input+'B/'
        travel_dataset=travel_dataset+'B/'
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
    else :
        print("problem color understand")
            
    if champ==champs[0]:
        travel_dataset=travel_dataset+'BF/'
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
    elif champ==champs[1]:
        travel_dataset=travel_dataset+'DF/'
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
    elif champ==champs[2]:
        travel_dataset=travel_dataset+'RAW_'+str(raw)+'/'
        if not os.path.exists(travel_dataset):
            os.mkdir(travel_dataset)
    else:
        print("problem champ understand")
        
    travels=[travel_input, travel_labeled, travel_dataset, travel_para]
    
    if champ==champs[2] and color==colors[1] and raw==0:
        segmentation='calcul'
    else:
        segmentation='extract'
        
    #extraction images and choice of image.        
    #extraction image in folder and create list of travel.
    images=[]
    file=os.listdir(travel_input)
    for i, chemin in enumerate(file):
        if '.tif' in chemin and champ==champs[2]:
            images.append(travel_input+'/'+chemin)
        elif champ==champs[1] and i%3==1 and ".bmp" in chemin:
            images.append(travel_input+'/'+chemin)
        elif champ==champs[0] and i%3==0 and ".bmp" in chemin:
            images.append(travel_input+'/'+chemin)
        elif '9994A324582' in chemin:
            print(chemin)
            
    return images, para, segmentation, travels