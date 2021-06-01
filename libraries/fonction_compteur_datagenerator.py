# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:57:14 2019

@author: gourgue
"""
#%%
import os
import time
from fonction_compteur_background import filtre_morpho

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

from skimage.measure    import regionprops
from skimage.morphology import watershed, label, black_tophat, disk
from skimage.io         import imsave

import pandas as pd
#%%
#cut a mask for watershed
def decoupe_mask(image, verbose=True, cells_mean=60):
    """decoupe_mask : delete more finely the background.
    return image with background is 0
    image   : input image
    verbose : display process
    """
    deb=time.time()
    markers=np.zeros_like(image)
    mask=filtre_morpho(image, verbose=verbose, cells_mean=cells_mean)
    markers[mask==0]=1
    markers[mask>0]=2
    
    if verbose=='all':
        plt.figure()
        plt.title('image')
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
        
        plt.figure()
        plt.title('mask')
        plt.imshow(mask, cmap='gray')
        
        plt.figure()
        plt.title('markers')
        plt.imshow(markers, cmap='gray')

    segmentation = watershed(image, markers)
    if verbose=='all':
        plt.figure()
        plt.title('segmentation binaire')
        plt.imshow(segmentation, cmap='gray')
        
    segmentation = ndi.binary_fill_holes(segmentation - 1)
    image_filtre=image.copy()
    image_filtre[segmentation]=0
    image_mask=image.copy()
    image_mask[~segmentation]=0
    if verbose=='all':
        plt.figure()
        plt.title("segmentation sans background")
        plt.imshow(segmentation, cmap='gray')

        plt.figure()
        plt.title('filtre')
        plt.imshow(image_filtre, cmap='gray', vmin=0, vmax=255)
        plt.colorbar()

        plt.figure()
        plt.title('mask')
        plt.imshow(image_mask, cmap='gray', vmin=0, vmax=255)
        plt.colorbar()
    fin=time.time()
    print(fin-deb)
    return image_mask
#%%

# ré éthiquetage
def fusion_label( region, labeled, cells_mean):
    """ fusion label : delete or fusion two region if is stick. (is not use)
    region     : is concerne
    labeled    : matrix with labeled
    cells_mean : diameter cells
    
        """
    x,y=region.centroid
    coords=region.coords
    boxe=labeled[max(int(x-cells_mean),0):min(int(x+cells_mean),labeled.shape[0]-1),
               max(int(y-cells_mean),0):min(int(y+cells_mean),labeled.shape[1]-1)]>0
    new_labels=label(boxe,neighbors=8)
    coords[:,0]=coords[:,0]-max(int(x-cells_mean),0)
    coords[:,1]=coords[:,1]-max(int(y-cells_mean),0)
    try :
        new_label =new_labels[coords[0,0],coords[0,1]]
    except :
        print("cas bizare")
        new_label = None
    for region_new in regionprops(new_labels):
        if region_new.label==new_label and region_new.area < 2500: #taille deux cellules
                if region_new.area>region.area:
                    max_label=np.max(labeled)
                    new_coords=region_new.coords
                    labeled[new_coords[:,0]+max(int(x-cells_mean),0),new_coords[:,1]+max(int(y-cells_mean),0)]=\
                    max_label+1
                    
                    return labeled
                else:
                    return region
#%%
#test premier filtre
def test_label(labeled, cells_mean, fusion=False, verbose=True):
    """ test_label : test the size of region and valide or not if is to small
    labeled    : matrix with labeled
    cells_mean : diameter of cells
    fusion     : use fusion fonction for delete or fusion small region
    verbose    : display process
    """
    cells_petite=[]
    for region in regionprops(labeled):
        if region.area>1000:
            if verbose=='all':
                print("area valide :"+str(region.area)+" pour region :"+str(region.label))
        else :
            if verbose=='all':
                print("area trop petit :"+str(region.area)+' pour la '+str(region.label))
            cells_petite.append(region)
        if region.equivalent_diameter >40:
            if verbose=='all':
                print("dimaetre equibalent valide pour region :"+str(region.label))
        else:
            if verbose=='all':
                print("dimaetre equibalent trop petit :"+str(region.equivalent_diameter)+" pour la "+str(region.label))
            
        if region.major_axis_length>50:
            if verbose=='all':
                print("grand axe valide :"+str(region.label))
        else:
            if verbose=='all':
                print("grand axe trop petit :"+str(region.major_axis_length)+" pour la "+str(region.label))
    if fusion==True:
        for region in cells_petite: 
            sortie=fusion_label(region, labeled,cells_mean)   
            if type(sortie)==np.ndarray:
                if verbose=='all':
                    print("cellule fusionné")
            else:
                labeled[region.coords[:,0],region.coords[:,1]]=0
                if verbose=='all':
                    print("zone supprimé")
        return labeled
    else :
        return labeled
#%%
def sauvegarde_imagette(image,zoro, classe,coords_para,coords_distrac,raw,name, cells_mean=60, size1=71,size2=71, 
                        travel_output=os.getcwd()+"_output/", tophat=False):
    """
    sauvegarde_imagette : save segmente cells with label if we have coordonate of parasite.
    image           : input image
    classe          : list of region segmented
    coords_para     : tuple or array with coordonate of all parasite.
    size1           : heigth of small picture
    size2           : width small picture
    travel_output   : folder output dataset
    """
    debut=time.time()
    #    test debug
    list_label=[]
    count=0
    #existence of folder
    if not os.path.exists(travel_output):
       os.mkdir(travel_output)
       #!mkdir -p travel_output
    #existence of folder tophat
    if tophat:
        travel_tophat=travel_output[:-1]+'_tophat/'
        if not os.path.exists(travel_tophat):
          os.mkdir(travel_tophat)
          #!mkdir -p travel_output
    #tranforme coords_para
    if type(coords_para)==np.ndarray:
        list_para=list(coords_para)
    elif coords_para is False:
        pass
    else:
        if 2 in coords_para.shape and len(coords_para.shape)==2:
            if coords_para.shape[0]==2:
                coords_para=coords_para.T
                list_para=list(coords_para)
        else:
            print("coords_para n'est pas à la bonne taille\ncoords_para shape=",coords_para.shape)
            return None
    if coords_distrac is False:
        pass
    else:
        list_distrac=list(coords_distrac)
    
    #tophat image
    if tophat:
        black_para=black_tophat(zoro, selem=disk(5))
        black_para[black_para.mask]=0
    
    for region in classe:
        infected=False
        distrac =False
        taille=image[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]].shape
        
        dx0 = region.bbox[0]
        dx1 = region.bbox[2]
        dy0 = region.bbox[1]
        dy1 = region.bbox[3]
        
        if(taille[0]%2!=0):
          dx1=dx1+1
        if(taille[1]%2!=0):
          dy1=dy1+1
        taille=image[dx0:dx1,dy0:dy1].shape
        
        if taille[0]>size1 or taille[1]>size2:
            print('problème de taille')
        else:
            image_sortie=np.zeros([size1,size2], dtype='uint8')
            if tophat:
                image_tophat=np.zeros_like(image_sortie)
            if region.equivalent_diameter>cells_mean*9/16:
                xc,yc=region.centroid
                x=int(xc-region.bbox[0])
                y=int(yc-region.bbox[1])
                center1=int(size1/2)
                center2=int(size2/2)
                begin_x=center1-x
                begin_y=center2-y
                coords = region.coords
                #test de depassement
                if np.max(coords[:,0]-region.bbox[0]+begin_x)>size1-1:
                    recalage_x=np.max(coords[:,0]-region.bbox[0]+begin_x)-size1+1
                    begin_x=begin_x-recalage_x
                if np.max(coords[:,1]-region.bbox[1]+begin_y)>size2-1:
                    recalage_y=np.max(coords[:,1]-region.bbox[1]+begin_y)-size2+1
                    begin_y=begin_y-recalage_y
                
                diffx1 = diffx2 = int((size1 - taille[0] )/2)
                diffy1 = diffy2 = int((size2 - taille[1] )/2)

                maxx,maxy = image.shape
                
                if(dx0-diffx1<0):
                  diffx2 = diffx2 + (dx0-diffx1)*(-1)
                  diffx1 = diffx1 - (dx0-diffx1)*(-1)
                if(dx1+diffx2 > maxx):
                  diffx1 = diffx1 + (maxx- dx1-diffx2)*(-1)
                  diffx2 = diffx2 - (maxx- dx1-diffx2)*(-1)
                if(dy0-diffy1<0):
                  diffy2 = diffy2 + (dy0-diffy1)*(-1)
                  diffy1 = diffy1 - (dy0-diffy1)*(-1)
                if(dy1+diffy2 > maxy):
                  diffy1 = diffy1 + (maxy- dy1-diffy2)*(-1)
                  diffy2 = diffy2 - (maxy- dy1-diffy2)*(-1)
                  
                #imagette image
                # image_sortie[coords[:,0]-region.bbox[0]+begin_x, coords[:,1]-region.bbox[1]+begin_y]=\
                # image[coords[:,0],coords[:,1]]
                
                image_sortie[center1-int(size1/2):center1+int(size1/2), center2-int(size2/2):center2+int(size2/2)]=\
                image[dx0-diffx1:dx1+diffx2,dy0-diffy1:dy1+diffy2]
                plt.imshow(image,cmap='gray')
                plt.show()

              
                #imagette tophat
                if tophat:
                    image_tophat[coords[:,0]-region.bbox[0]+begin_x, coords[:,1]-region.bbox[1]+begin_y]=\
                black_para[coords[:,0],coords[:,1]]
                
                #title='('+str(int(round(xc)))+','+str(int(round(yc)))+')'+'_'+str(region.label)+'.png'
                title=str(raw)+'.png'
                if coords_para is False:
                    if len(list_para)>0:
                        if len(list_para[0])==0:
                            pass
                        else:
                            for i in range(len(list_para)-1,-1,-1):
                                if list_para[i][0] in coords[:,0]:
                                    indice=np.where(coords[:,0]==list_para[i][0])[0]
                                    resultat=np.where(coords[indice,1]==list_para[i][1])[0]
                                    if len(resultat)!=0:
                                        place=indice[resultat][0]
                                        if len(resultat)==1:               
                                            list_para.pop(i)
                                            infected=True
                                        else :
                                            print("problem a coords parasite is more than 1 time in a cells")
                                            print("coords = ",place)

              
                # if infected:
                #     #pas de distract
                #     pass
                # else:
                #     if coords_distrac is False:
                #       if len(list_distrac)>0:
                #         if len(list_distrac[0])==0:
                #             pass
                #     else:
                #         for i in range(len(list_distrac)-1,-1,-1):
                #             if list_distrac[i][0] in coords[:,0]:
                #                 indice=np.where(coords[:,0]==list_distrac[i][0])[0]
                #                 resultat=np.where(coords[indice,1]==list_distrac[i][1])[0]
                #                 if len(resultat)!=0:
                #                     place=indice[resultat][0]
                #                     if len(resultat)==1:               
                #                         list_distrac.pop(i)
                #                         distrac=True
                #                         print("dis")
                #                     else :
                #                         print("problem a coords distractor is more than 1 time in a cells")
                #                         print("coords = ",place)

                
                if infected:
                    title='_infected_'+title
                else:
                    if distrac:
                        title='_distrac_'+title
                    else:
                        title='_healthy_'+title

                imsave(travel_output+str(count)+title, image_sortie)
                print(title)
                count=count+1
                list_label.append(region.label)
                if tophat:
                    imsave(travel_tophat+title, image_tophat, check_contrast=False)
                
    fin=time.time()
    print(fin-debut)
    if len(list_para)>0:
        print("il reste des parasites non detecter")
        print("longueur de liste para =",len(list_para))
        return list_para, list_label
    else:
        return None
                                       

#%%
def extract_para(travel_para):
    coords_para=pd.read_csv(travel_para, sep=';')
    if coords_para.shape[1]==0:
        coords_para=[[],[]]
    return coords_para

#%%
def extract_imagette(image, labeled, coords_para, coords_distrac,cells_mean=60, size1=71, size2=71, 
                     travel_output=os.getcwd()+"_output/", tophat=False, zoro=False):
    debut=time.time()
    #existence of folder
    if not os.path.exists(travel_output):
       os.mkdir(travel_output)
       
    #existence of folder tophat
    if tophat:
        travel_tophat=travel_output[:-1]+'_tophat/'
        if not os.path.exists(travel_tophat):
           os.mkdir(travel_tophat)
       
    #tranforme coords_para
   
    if coords_para is False:
        pass
    else:
         if type(coords_para)==np.ndarray:
            pass
         else:
            if 2 in coords_para.shape and len(coords_para.shape)==2:
                if coords_para.shape[0]==2:
                    coords_para=coords_para.T
            else:
                print("coords_para n'est pas à la bonne taille\ncoords_para shape=",coords_para.shape)
                return None
         list_para=list(coords_para)
    if coords_distrac is False:
        list_distrac=[]
        pass
    else:
        list_distrac =list(coords_distrac)
    
    #tophat image
    if tophat:
        black_para=black_tophat(zoro, selem=disk(5))
        black_para[black_para.mask]=0
    
    for region in regionprops(labeled):
        infected=False
        distrac=False
        taille=image[region.bbox[0]:region.bbox[2],region.bbox[1]:region.bbox[3]].shape
        if taille[0]>size1 or taille[1]>size2:
            print("\n taille[0], [1]=",taille[0],",",taille[1])
            print("\n size1", size1)
           
            print('problème de taille')
        else:
            image_sortie=np.zeros([size1,size2], dtype='uint8')
            image_tophat=np.zeros_like(image_sortie)
            if region.equivalent_diameter>cells_mean*2/3:
                xc,yc=region.centroid
                x=int(xc-region.bbox[0])
                y=int(yc-region.bbox[1])
                center1=int(size1/2)
                center2=int(size2/2)
                begin_x=center1-x
                begin_y=center2-y
                coords = region.coords
                #test de depassement
                if np.max(coords[:,0]-region.bbox[0]+begin_x)>size1-1:
                    recalage_x=np.max(coords[:,0]-region.bbox[0]+begin_x)-size1+1
                    begin_x=begin_x-recalage_x
                if np.max(coords[:,1]-region.bbox[1]+begin_y)>size2-1:
                    recalage_y=np.max(coords[:,1]-region.bbox[1]+begin_y)-size2+1
                    begin_y=begin_y-recalage_y
                    
                #imagette image
                image_sortie[coords[:,0]-region.bbox[0]+begin_x, coords[:,1]-region.bbox[1]+begin_y]=\
                image[coords[:,0],coords[:,1]]
                
                #imagette tophat
                if tophat:
                    image_tophat[coords[:,0]-region.bbox[0]+begin_x, coords[:,1]-region.bbox[1]+begin_y]=\
                black_para[coords[:,0],coords[:,1]]
                
#               title=str(region.label)+'_'+str(int(round(xc)))+'x'+str(int(round(yc)))+'.png'
                print(title)
                title=str(raw)+'.png'
                if coords_para is False:
                    pass
                else:
                    if len(list_para)>0:
                        if len(list_para[0])==0:
                            pass
                        else:
                            for i in range(len(list_para)-1,-1,-1):
                                if list_para[i][0] in coords[:,0]:
                                    indice=np.where(coords[:,0]==list_para[i][0])[0]
                                    resultat=np.where(coords[indice,1]==list_para[i][1])[0]
                                    if len(resultat)!=0:
                                        place=indice[resultat][0]
                                        if len(resultat)==1:               
                                #        print(i,"est dans la matrice 2 en",place, "mat2["+str(place)+",:]=",matrice2[place])
                                            list_para.pop(i)
                                            infected=True
                                        else :
                                            print("problem a coords parasite is more than 1 time in a cells")
                                            print("coords = ",place)
#                        print(valeur,"en matrice 2 à",place)
                if infected:
                    #pas de distracteur
                    pass
                else:
                    if coords_distrac is False:
                        pass
                    else:
                        for i in range(len(list_distrac)-1,-1,-1):
                            if list_distrac[i][0] in coords[:,0]:
                                indice=np.where(coords[:,0]==list_distrac[i][0])[0]
                                resultat=np.where(coords[indice,1]==list_distrac[i][1])[0]
                                if len(resultat)!=0:
                                    place=indice[resultat][0]
                                    if len(resultat)==1:               
                            #        print(i,"est dans la matrice 2 en",place, "mat2["+str(place)+",:]=",matrice2[place])
                                        list_distrac.pop(i)
                                        distrac=True
                                    else :
                                        print("problem a coords parasite is more than 1 time in a cells")
                                        print("coords = ",place)
    #                        print(valeur,"en matrice 2 à",place)
                
                
                if infected:
                    title='infected_'+title
                else:
                    if distrac:
                        title = 'distrac_'+title
                    else:
                        title='healthy_'+title
    #    plt.close("all")
    #    plt.figure()
    #    plt.imshow(image_sortie)
                imsave(travel_output+title, image_sortie)
                if tophat:
                    imsave(travel_tophat+title, image_tophat, check_contrast=False)
    fin=time.time()
    print(fin-debut)
#    print(coords_para)
    if not coords_para is False:
        if len(list_para)>0:
            print("il reste des parasites non detecter")
            print("longueur de liste para =",len(list_para))
            return list_para
    else:
        return None
