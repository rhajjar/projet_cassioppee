# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:13:10 2019

@author: gourgue
"""
#%%
import numpy as np

from scipy import ndimage as ndi
from scipy.io import savemat

import matplotlib.pyplot as plt

from fonction_compteur_affiche       import plot
from fonction_compteur_datagenerator import test_label

from skimage.morphology import watershed, disk, black_tophat, white_tophat
from skimage.feature    import canny
from skimage.transform  import hough_circle, hough_circle_peaks
from skimage.measure    import regionprops
from skimage.filters    import threshold_otsu ,rank
from skimage.draw       import circle_perimeter, circle
from skimage.io         import imsave

import time

import os 

import pandas as pd

#%%
def test_region(labeled, cells_mean=60, threshold='hough_iter', bord=False):
    """ test region is function for determine if a label is a cells or a cluster.
    labeled    : matrix with labels
    cells_mean : diameter cells
    threshold  : hough_iter for the global image and hough_cluster for the semgentation cells
    bord       : for accept more cells in the edges of picture. 
    """
    classe=[]
    amas_cells=[]
    for region in regionprops(labeled):
        if threshold=='hough_iter':
            if region.area>(((cells_mean/2)*0.94)**2)*np.pi or region.major_axis_length>cells_mean*1.34:
                #tolérance au bord 
#                print("tolérance au bord")
#                print(region.area, region.major_axis_length)
                if bord:
                    #si bord
                    if region.centroid[0]<cells_mean or region.centroid[0]>labeled.shape[0]-cells_mean or\
                       region.centroid[1]<cells_mean or region.centroid[1]>labeled.shape[1]-cells_mean:
                           #trop gros
                           if region.equivalent_diameter>cells_mean*1.1 and region.major_axis_length >cells_mean*1.5:
                               amas_cells.append(region)
                           #accepter
                           else:
                               classe.append(region)
                    #pas bord refuser
                    else:
                        amas_cells.append(region)
                    
                #pas de tolérence au bord    
                else:
                    amas_cells.append(region)
            elif region.area>(((cells_mean/2)*0.595)**2)*np.pi:
#                print("region ajoutées")
                classe.append(region)
            else:
#                print("region trop petite suppression")
#                print("taille custer :",(((cells_mean/2)*0.94)**2)*np.pi, "ou", cells_mean*1.34)
#                print("taille réel :",region.area, "et",  region.major_axis_length)
#                print("taille requis pour une cellules :", (((cells_mean/2)*0.595)**2)*np.pi)
#                print("taille réel :", region.area)
                coords=region.coords
                labeled[coords[:,0],coords[:,1]]=0
                
                
        elif threshold=='hough_cluster':
            if region.area>(((cells_mean/2)*0.94)**2)*np.pi or region.major_axis_length>cells_mean*1.34:
                #tolérance au bord 
                if bord:
                    #si bord
                    if region.centroid[0]<cells_mean or region.centroid[0]>labeled.shape[0]-cells_mean or\
                       region.centroid[1]<cells_mean or region.centroid[1]>labeled.shape[1]-cells_mean:
                           #trop gros
                           if region.equivalent_diameter>cells_mean*1.1 and region.major_axis_length >cells_mean*1.5:
                               amas_cells.append(region)
                           #accepter
                           else:
                               classe.append(region)
                    #pas bord refuser
                    else:
                        amas_cells.append(region)
                    
                #pas de tolérence au bord    
                else:
                    amas_cells.append(region)
            elif region.area>(((cells_mean/2)*0.595)**2)*np.pi:
                classe.append(region)
            #repeche des petites régions
            elif region.area >(((cells_mean/2)*0.515)**2)*np.pi and region.convex_area >(((cells_mean/2)*0.595)**2)*np.pi:
                classe.append(region)
                
            else:
                print("problème cellule coupé en deux")

    return classe, amas_cells    


        

#%%    
def detect_para (image, zoro, verbose=True, title="image parasite", thres=10, save=None, champ='bright'):   
    """ detect_para function for detect_parasite. for the moment is succesfull only the CAT-01 image.
    
    image   : image d'origine.
    zoro    : masque du fond
    verbose : display process
    title   : title of image
    thres   : number of times Otsu thresholding
    save    : if you want to save the coordonate in dictionnary file
    """
    
    if champ=='bright':
        morpho_para=black_tophat(zoro, selem=disk(5))
    elif champ=='dark':
        morpho_para=white_tophat(zoro, selem=disk(5))
    else:
        print("champ no reconnize. possible value : 'bright','dark'\nbut champ=",champ)
    morpho_para[morpho_para.mask]=0
    thres_local_3=rank.otsu(morpho_para, disk(3))
    thres_para=threshold_otsu(thres_local_3)
    
    coords_para=np.where(thres_local_3>thres_para*thres)
    labeled_para, nb_para = ndi.label(thres_local_3>thres_para*thres)
    #visualisation parasite
    image_para_fill=np.zeros([image.shape[0],image.shape[1],3],dtype='uint8')
    image_para_fill[:,:,0]=image
    image_para_fill[:,:,1]=image
    image_para_fill[:,:,2]=image
    image_para_perimeter=image_para_fill.copy()
    #color image
    
    for para in regionprops(labeled_para):
        r,c=para.centroid
        radius=int(para.equivalent_diameter/2)
    
        rect_y, rect_x=circle_perimeter(int(r),int(c),radius*2,shape=image.shape)
        image_para_perimeter[rect_y,rect_x,:]=[255,0,0]
        rect_y, rect_x=circle(int(r),int(c),radius*2, shape=image.shape)
        image_para_fill[rect_y,rect_x,:]=[255,0,0]
        
    if verbose=='all':
        plot(morpho_para,'filtre')
        plot(thres_local_3,'para thres')
        plot(thres_local_3>thres_para*thres,'thres_para')
    if verbose=='all' or verbose==True:    
        plt.figure()
        plt.title(title+" edges")
        plt.imshow(image_para_perimeter)
        plt.figure()
        plt.title(title+" fill")
        plt.imshow(image_para_fill)
    if save is None:
        pass
    elif type(save)==str:
        dico={}
        dico.update({title+" fill":image_para_fill})
        dico.update({title+" edges":image_para_perimeter})
        savemat(save+title, dico)
        
    
    markers_para, nb_para=ndi.label(thres_local_3>thres_para*thres)
    coords_para_save=[]
    for region in regionprops(markers_para):
        coords_para_save.append(region.centroid)
    
    coords_para_save=pd.DataFrame(np.array(coords_para_save))
    coords_para_save.to_csv(os.getcwd()+'/coords_para.csv')
    
    
    return coords_para        

    
        
#%%
def break_cluster(image, amas_cells, labeled, cells_mean=60, verbose=False, title='', condition=["nombre",10],
                  ray=0.2, exemple=[False]):
    
    """ break cluster is a method for separate few cells are stick. 
    the method function in Hough circular transform. we try to separate the cells with small markers
    in the center of cells and if not function we retry with a bigger markers in the center and if not
    function we return the cluster
    
    image      : orignal image
    amas_cells : list of region with cluster
    labeled    : matrix with labeled of pixel
    cells_mean : diamter of cells
    verbose    : display process
    title      : title of image
    condition  : list for the condition for hough circular tranforme threshold. Argument first 'seuil' 
    for stop marker of hough circular transform by threshold by accumulator. The second argument is the
    threshold. Argument first 'nombre' for stop marker of hough circular transform by number of markers. 
    The second argument is the number of center to concerve.
    ray        : the proportion of marker by hough circular detection. by the default radius of the marker 
    is 20% of the radius detected by hough circular transform."""
#%%
    for l in amas_cells:
        #%%
        #extraction picture
        boxe, labels_alone=l.bbox, l.label
        labeled_rogner=labeled[boxe[0]:boxe[2],boxe[1]:boxe[3]].copy()
        mask_rogner=np.ma.masked_where(labeled_rogner!=labels_alone,labeled_rogner)
        image_rogner=np.ma.masked_array(image[boxe[0]:boxe[2],boxe[1]:boxe[3]],mask_rogner.mask).copy()
        
        #création marker
        markers=np.zeros_like(image_rogner)
        markers=np.array(markers, dtype='uint16')
                
        if verbose=='all':
            plt.figure()
            plt.title('labeled_rogner')
            plt.imshow(labeled_rogner)
            plt.figure()
            plt.title('mask_rogner')
            plt.imshow(mask_rogner)
            plt.figure()
            plt.title('image_rogner')
            plt.imshow(image_rogner)  
        
        if exemple[0]:
            img_cr=(labeled_rogner -labeled_rogner.min())/(labeled_rogner.max()-labeled_rogner.min())
            labeled_rogner_save=img_cr*255
            del(img_cr)
            labeled_rogner_save=np.array(labeled_rogner_save, dtype='uint8')
            imsave(exemple[1]+str(exemple[2])+'.png', labeled_rogner_save)
            exemple[2]+=1
            img_cr=(mask_rogner -mask_rogner.min())/(mask_rogner.max()-mask_rogner.min())
            mask_rogner_save=img_cr*255
            del(img_cr)
            mask_rogner_save=np.array(mask_rogner_save, dtype='uint8')
            imsave(exemple[1]+str(exemple[2])+'.png', mask_rogner_save)
            exemple[2]+=1
            img_cr=(image_rogner -image_rogner.min())/(image_rogner.max()-image_rogner.min())
            image_rogner_save=img_cr*255
            del(img_cr)
            image_rogner_save=np.array(image_rogner_save, dtype='uint8')
            imsave(exemple[1]+str(exemple[2])+'.png', image_rogner_save)
            exemple[2]+=1
    
        min_radius=int(cells_mean/3)
        max_radius=int(cells_mean/1.5)
        #hough sur image d'origine
        image_hough=image_rogner.copy()

        #hough sur image gradiant
        image_hough[image_hough.mask]=0
        edges_canny=canny(image_hough, sigma=5, low_threshold=0, high_threshold=5.2)
        
        if verbose=='all':
            plot(image_hough, 'image avec 0')
            plot(edges_canny,'canny')
        if exemple[0]:
            try:
                img_cr=(edges_canny -edges_canny.min())/(edges_canny.max()-edges_canny.min())
                edges_canny_save=img_cr*255
                del(img_cr)
                edges_canny_save=np.array(edges_canny_save, dtype='uint8')
                imsave(exemple[1]+str(exemple[2])+'.png', edges_canny_save)
                exemple[2]+=1
            except:
                plot(edges_canny,'canny')                

        if image_hough.dtype!='uint8':
            image_hough=image_hough*255/image_hough.max()
            image_hough=np.array(image_hough, dtype='uint8')
        
        if verbose=='all':
            plot(image_hough, 'image_hough before circle')
            
        # Detect two radii
        hough_radii = np.linspace(min_radius, max_radius, 10)
        hough_radii = np.array(hough_radii, dtype='uint8')
        hough_res = hough_circle(edges_canny, hough_radii)
        try:
            accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=int(cells_mean/2), 
                                                   min_ydistance=int(cells_mean/2))
        except:
            #debogue
            from scipy.io import savemat
            dico={'image':image, 'image_r':image_rogner, 'edges':edges_canny, 'condition':condition,"hough":hough_res}
            name_dico=str(round(condition[1],4))
            print("name_dico:",name_dico)
            savemat(name_dico, dico)
            accums=[]
            cx=[]
            cy=[]
            radii=[]
        
        if condition[0]=='seuil':
            condition_accums=condition[1]
            nb_cercle=np.sum(accums>condition_accums)
            accums=accums[:nb_cercle]
            cx=cx[:nb_cercle]
            cy=cy[:nb_cercle]
            radii=radii[:nb_cercle]
        elif condition[0]=='nombre':
            condition[1]=int(5*round(l.area/(np.pi*(cells_mean/2*0.84)**2)))
            accums=accums[:condition[1]]
            cx=cx[:condition[1]]
            cy=cy[:condition[1]]
            radii=radii[:condition[1]]

        for center_y, center_x, radius in zip(cy, cx, radii):
            
            circy, circx = circle(center_y, center_x, int(radius*ray), shape=image_hough.shape)
            markers[circy, circx] = 2
        
        
        markers[mask_rogner.mask]=0
                        
                        
        if verbose=='all':
            plot(markers,'markers')
        if exemple[0]:
                img_cr=(markers -markers.min())/(markers.max()-markers.min())
                markers_save=img_cr*255
                del(img_cr)
                markers_save=np.array(markers_save, dtype='uint8')
                imsave(exemple[1]+str(exemple[2])+'.png', markers_save)
                exemple[2]+=1                 

        markers = ndi.label(markers)[0]
        
        markers=test_distance(markers, verbose=verbose, threshold=500)   
        #attribution label for origin image
        max_label=np.max(np.unique(labeled))
        markers[markers>0]=markers[markers>0]+max_label
            
        if verbose=='all':
            plot(markers, 'markers with big label')
        if exemple[0]:
            img_cr=(markers -markers.min())/(markers.max()-markers.min())
            markers_save=img_cr*255
            del(img_cr)
            markers_save=np.array(markers_save, dtype='uint8')
            imsave(exemple[1]+str(exemple[2])+'.png', markers_save)
            exemple[2]+=1   
            
        labels = watershed(image_rogner, markers, mask=~image_rogner.mask)
        
        if verbose=='all':
            plt.figure()
            plt.title("labels new")
            plt.imshow(labels)
        if exemple[0]:
            img_cr=(labels -np.unique(labels)[1]+1)/(labels.max()-np.unique(labels)[1]+1)
            labels_save=img_cr*255
            labels_save[labels_save<1]=0
            del(img_cr)
            labels_save=np.array(labels_save, dtype='uint8')
            imsave(exemple[1]+str(exemple[2])+'.png', labels_save)
            exemple[2]+=1     
        #test decoupe
        decoupe=True
        area_max=0
        label_max=0
        region_petite=[]
        for region in regionprops(labels):
            if region.area>area_max:
                area_max=region.area
                label_max=region.label
            if region.convex_area<(((cells_mean/2)*0.595)**2)*np.pi:
                decoupe=False
                region_petite.append(region)
                
        if decoupe is False:
            for region in region_petite:
                labels[region.coords[:,0],region.coords[:,1]]=label_max
                
            #transfert label
            labeled_trans=labeled[boxe[0]:boxe[2],boxe[1]:boxe[3]]
            labeled_trans[labels>0]=labels[labels>0]
            classe, amas_cells_local=test_region(labels, cells_mean=cells_mean, threshold='hough_cluster', bord=False)
            if ray<0.5 and len(amas_cells_local)>0:
                labeled_rogner =break_cluster(image_rogner, amas_cells_local, labels,verbose=verbose,ray=ray*2, exemple=exemple)
                classe, amas_cells_local=test_region(labeled_rogner, cells_mean=cells_mean, threshold='hough_cluster', bord=False)
                if len(amas_cells_local)==0:
                    labeled_trans=labeled[boxe[0]:boxe[2],boxe[1]:boxe[3]]
                    labeled_trans[labeled_rogner>0]=labeled_rogner[labeled_rogner>0]
    

            #recommencer avec les cercles. 
        if decoupe==True:
            #transfert label
            labeled_trans=labeled[boxe[0]:boxe[2],boxe[1]:boxe[3]]
    #        print(labeled_trans.shape,labels.shape)
            labeled_trans[labels>0]=labels[labels>0]
            
            if verbose=='all':
                plot(labeled_trans, "label transfert")
                plot(labeled, "labeled complet image")
            
   
    return labeled#, amas_cells

#%%
    

def test_distance(markers, verbose=True,threshold=200,size=None):
    """detect if to center is too near than seuil delet the smaller
    
    markers     : image input
    verbose     : if you want display process
    threshold   : the sqaure distance into center
    size        : the minimum size of area of the retained marker """
    regions = regionprops(markers)
    for num1 in range(len(regions)-1):
        region1=regions[num1]
        if size is not None:
            if region1.area >size:
                coords=region1.coords
                markers[coords[:,0],coords[:,1]]=0
            else :
                for region2 in regions[num1+1:] :
                    yi,xi=region1.centroid
                    yj,xj=region2.centroid
                    if region1.label==region2.label:
                        pass
                    elif (xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)<threshold:
                        if region1.area<region2.area:
                            if verbose=='all':
                                print("on supprime",region1.label)
                            coords=region1.coords
                         
                        else:
                            if verbose=='all':
                                print("on supprime", region2.label)
                            coords=region2.coords
                            
                        markers[coords[:,0],coords[:,1]]=0
        elif size is None:
            for region2 in regions[num1+1:] :
                yi,xi=region1.centroid
                yj,xj=region2.centroid
                if region1.label==region2.label:
                    pass
                elif (xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)<threshold:
                    if region1.area<region2.area:
                        if verbose=='all':
                            print("on supprime",region1.label)
                        coords=region1.coords
                     
                    else:
                        if verbose=='all':
                            print("on supprime", region2.label)
                        coords=region2.coords
                        
                    markers[coords[:,0],coords[:,1]]=0
            
    return markers


#%%
def Hough_by_thres(image, zoro, cells_mean=60, verbose=True,condition=['seuil',1,0.9,1], 
                   edges=None,labeled_list=[], exemple=[False] ):
    """ methode with Hough circular tranfrom. 
    we use a Hough for detect the center of cells after we do a watershed for fill cells.
    after we detect cluster and small segmentation. we delete small zone and break the cluster. 
    if the cluster is not break is delete. 
    
    image       : image orignal
    zoro        : image with mask background
    cells_mean  : diameter of cells in pixel
    verbose     : if you want display the step of process
    condition   : list with fist argument 'seuil' or 'nombre' if you want the hough transform stop by a thres
    or a number of center
    edges       : image of contour (canny here)
    labeled_list: the output is a list but the iteration of fonction need to input the list for add the labeled
    """
#%%
    deb=time.time()
    if verbose=='all':
        plot(image,'image origine')
    if exemple[0]:
        imsave(exemple[1]+str(exemple[2])+'.png', image)
        exemple[2]+=1
    markers=np.zeros_like(image)
    if edges is None:
        #first passage
        edges=np.zeros_like(image)
        edges=canny(zoro, sigma=7, low_threshold=0.2, high_threshold=5.2)
    if verbose=='all':
        plot(edges,'egdes')
    if exemple[0]:
        try:
            imsave(exemple[1]+str(exemple[2])+'.png', edges)
            exemple[2]+=1
        except:
            plot(edges,'egdes')
        
    image_hough=image.copy()
    image_hough=(image_hough-image_hough.min())/(image_hough.max()-image_hough.min())*243+1
    image_hough=np.array(image_hough, dtype='uint8')
    min_radius=int(cells_mean/3)
    max_radius=int(cells_mean/1.5)

    
        
    #3 detection des cellules
    #Hough circular transform
    hough_radii = np.linspace(min_radius, max_radius, 10)
    hough_radii = np.array(hough_radii, dtype='uint8')
    hough_res = hough_circle(edges, hough_radii)
    try :
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=int(cells_mean/2), 
                                               min_ydistance=int(cells_mean/2))
    except:
        #debogue
        #problème non reglé.
        from scipy.io import savemat
        dico={'image':image, 'zoro':zoro, 'edges':edges, 'condition':condition,"hough":hough_res}
        name_dico=str(round(condition[1],4))
        print("name_dico:",name_dico)
        savemat(name_dico, dico)
        
    
    if condition[0]=='seuil':
        maxi=accums.max()
        thres=maxi*condition[2]
        if thres==condition[1]:
#            return labeled_list
            thres=condition[1]*condition[2]
            condition[1]=thres
            condition[2]=condition[2]*condition[2]
        elif thres < condition[1]:
            condition[1]=thres
        elif thres >condition[1]:
            condition[1]=condition[1]*condition[2]
            print("il y a un problème")
        else :
            print("is not possible")
        nb_cercle=np.sum(accums>condition[1])
        accums=accums[:nb_cercle]
        cx=cx[:nb_cercle]
        cy=cy[:nb_cercle]
        radii=radii[:nb_cercle]
        
    edges_hough=edges.copy()
    delete_edges=np.zeros_like(image)
    sure_fg=np.zeros_like(image_hough)
    
    taille_list = len(labeled_list)
    if taille_list==0:
        mini=1
    elif taille_list>0:
        mini=condition[3]
    else:
        print("labeled_list bizarre")
    for center_y, center_x, radius in zip(cy, cx, radii):
        #markers cells center
        circy, circx = circle(center_y, center_x, int(radius*0.5), shape=image_hough.shape)
        sure_fg[circy, circx] = 1
#        circy, circx = circle_perimeter(center_y, center_x, int(radius*1.2), shape=image_hough.shape)
        #for delete edges
        ciry, cirx = circle(center_y,center_x, radius+1,shape=image_hough.shape)
        delete_edges[ciry,cirx]=1
        
    
    #reéhiquetage
    markers,nb_cells=ndi.label(sure_fg)
    markers[markers>0]=markers[markers>0]+mini
    delete_edges, nb_cell= ndi.label(delete_edges)
#    print("markers:",markers.min(),markers.max())
            
    #condition of background    
    sure_bg=np.zeros_like(image_hough)+255
    sure_bg[image_hough>200]=0
    sure_bg[zoro.data==0]=0
    
    
    #4 markers
    if verbose=='all':
        plot(markers, 'cells markers')
        plot(sure_fg, 'sure fg')
        plot(delete_edges, 'edges delete')
        plot(image_hough, 'image avec contour')
        plot(sure_bg, 'back')

    # Marker labelling
    # Add one to all labels so that sure background is not 0, but 1
    markers[sure_bg==0] = 1
    if verbose=='all':
        plot(markers, "markers")
    if exemple[0]:
        imsave(exemple[1]+str(exemple[2])+'.png', markers)
        exemple[2]+=1


    #5 watershed
    image_hough_3D=np.zeros([image.shape[0],image.shape[1],3], dtype='uint8')
    image_hough_3D[:,:,0]=image_hough
    image_hough_3D[:,:,1]=image_hough
    image_hough_3D[:,:,2]=image_hough
    water='openCV'
    if water=='skimage':
        labeled = watershed(edges_hough, markers)
        
    elif water=='openCV':
#        import cv2
#        segmentation = cv2.watershed(image_hough_3D, markers)
        labeled = watershed(image_hough, markers)
    labeled[labeled==1]=0
    
    if verbose=='all':

        plt.figure()
        plt.title("marker")
        plt.imshow(markers, cmap='gray')

        plt.figure()
        plt.title("labeled")
        plt.imshow(labeled, cmap='gray')
    if verbose==True:
        plt.figure()
        plt.title("labeled")
        plt.imshow(labeled, cmap='gray')
    
    if exemple[0]:
        img_cr=(labeled-np.unique(labeled)[1]+1)/(labeled.max()-np.unique(labeled)[1]+1)
        labeled_save=img_cr*255
        labeled_save[labeled_save<1]=0
        del(img_cr)
        labeled_save=np.array(labeled_save, dtype='uint8')
        imsave(exemple[1]+str(exemple[2])+'.png', labeled_save)
        exemple[2]+=1   
    
    
    #test des labels trouvé pas de cluster
    classe, amas_cells=test_region(labeled, cells_mean=cells_mean, threshold='hough_iter', bord=True)
    #%%
    if len(amas_cells)>0:
        labeled=break_cluster(image, amas_cells, labeled, cells_mean=cells_mean, verbose=verbose, title='', 
                    condition=["nombre",10], ray=0.2, exemple=exemple)#, separate='hough circle', boxes=['ellipse','blue']
        classe, amas_cells=test_region(labeled, cells_mean=cells_mean, threshold='hough_cluster', bord=True)
        
        if len(amas_cells)>0:
            if verbose is True:
                print("efface cluster")
            for i in amas_cells:
                centre=i.centroid
                lab=delete_edges[int(centre[0]),int(centre[1])]
                labeled[i.coords[:,0],i.coords[:,1]]=0
                delete_edges[delete_edges==lab]=0
                    
    #for function recurence
    edges[delete_edges==1]=0    
    edges[labeled>0]=0
    image_suivante=image.copy()
    image_suivante[labeled>0]=0
    zoro.data[labeled>0]=0
    zoro.mask[labeled>0]=True
    labeled_list.append(labeled)
    
    fin=time.time()
    if verbose is True:
        print('time=',fin-deb)
    if condition[1]>0.1: #(cells_mean/60*0.2):
        if verbose is True:
            print("thres=",condition[1])
        #%%
        _=Hough_by_thres(image_suivante,zoro, cells_mean=cells_mean, verbose=verbose,condition=condition, 
                   edges=edges,labeled_list=labeled_list )
        
    fin=time.time()
    if verbose is True:
        print('time=',fin-deb)

#%%
    return labeled_list
def recon_image(labeled_list,cells_mean=60,verbose=True):
    """the Hough tranform generate a list of label. the recon_image add all label a one picture
    
    labeled_list : list of labeled to add
    cells_mean   : diamter cells
    verbose      : display image in process"""
    for i in range(len(labeled_list)-1):
        if verbose=='all':
            print(i, len(labeled_list))
        image1=labeled_list.pop(0)
        image2=labeled_list.pop(0)
        maxi=image2.max()
        image1[image1>0]=image1[image1>0]+maxi
        image2[image1>0]=image1[image1>0]
        image2=test_label(image2, cells_mean, False)
        labeled_list.insert(0,image2)
    return labeled_list

#%%
def abstrat_back(image, size=60):
    thres_local=rank.otsu(image,disk(size))
    if image.dtype=='uint8':
        image=np.array(image, dtype='uint16')
    image=image+thres_local.max()
    image=image-thres_local
    image=(image-image.min())/(image.max()-image.min())*255
    image=np.array(image,dtype='uint8')
    return image