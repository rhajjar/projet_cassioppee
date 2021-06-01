# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 12:09:31 2019

@author: gourgue
"""
#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.color      import label2rgb

import numpy as np

from skimage.draw import ellipse_perimeter, circle
#%%
def affiche(image, labeled,classe, title=None,boxes=False):
    """function for affiche image with edges and image with segmentation cells.
    return the figure, and the image of right for save. 
    image   : orignal image.
    labeled : matrix with labeled
    classe  : list with region of cells
    title   : title
    boxes   : display edges of cells. True for a rectangle, red. other cases is list.
            fist argument is box for rectangle or ellipse for ellipse.
            second argument is color.
    """
    image_label_overlay = label2rgb(labeled, image=image)
    fig1=plt.figure(title)
    axes=[]
    ax1=fig1.add_subplot(1,2,1, label="image with edge")
    axes.append(ax1)
    ax2=fig1.add_subplot(1,2,2, label="image with classes")
    axes.append(ax2)
#    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    axes[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    axes[0].contour(np.array(labeled, dtype=bool), [0.5], linewidths=1.2, colors='y')
    axes[0].set_title("image with edge")
    axes[1].imshow(image_label_overlay, interpolation='nearest')
    axes[1].set_title("image with classes")
    
    if title is not None:
        fig1.suptitle(title)
        
    if boxes==True:
        #par défaut boites rouge
        for region in classe:
            boxe=region.bbox
            rect = mpatches.Rectangle((boxe[1], boxe[0]), boxe[3] - boxe[1], boxe[2] - boxe[0],
                                              fill=False, edgecolor='red', linewidth=1)
            axes[1].add_patch(rect)
    elif type(boxes)==list:
        if len(boxes)==1:
            # cas de l'élispse
            if boxes[0]=='ellipse':
                for region in classe:
                    y,x =region.centroid
                    orientation=-region.orientation*180/np.pi
                    grand_axe=region.major_axis_length
                    petit_axe=region.minor_axis_length
                    hell=mpatches.Ellipse([y,x],grand_axe,petit_axe,angle=orientation,fill=False, edgecolor='red', linewidth=1)
                    axes[1].add_patch(hell)
            #cas boites par défaut rouge   
            elif boxes[0]=='box':
                for region in classe:
                    boxe=region.bbox
                    rect = mpatches.Rectangle((boxe[1], boxe[0]), boxe[3] - boxe[1], boxe[2] - boxe[0],
                                                      fill=False, edgecolor='red', linewidth=1)
                    axes[1].add_patch(rect)
            else :
                #cas couleurs par défaut boites
                for region in classe:
                    boxe=region.bbox
                    rect = mpatches.Rectangle((boxe[1], boxe[0]), boxe[3] - boxe[1], boxe[2] - boxe[0],
                                                      fill=False, edgecolor=boxes[0], linewidth=1)
                    axes[1].add_patch(rect)
        elif len(boxes)==2:
            #cas boites et couleur
            if boxes[0]=='box':
                for region in classe:
                    boxe=region.bbox
                    rect = mpatches.Rectangle((boxe[1], boxe[0]), boxe[3] - boxe[1], boxe[2] - boxe[0],
                                                      fill=False, edgecolor=boxes[1], linewidth=1)
                    axes[1].add_patch(rect)
            elif boxes[0]=='ellipse':
                for region in classe:
                    y,x =region.centroid
                    orientation=-region.orientation*180/np.pi
                    grand_axe=region.major_axis_length
                    petit_axe=region.minor_axis_length
                    hell=mpatches.Ellipse([x,y],grand_axe,petit_axe,angle=orientation,fill=False, edgecolor=boxes[1], linewidth=1)
                    axes[1].add_patch(hell)
        else:
            print('format de boxes non reconnu')
            print('boxes =',boxes)
    
            
    for a in axes:
        a.axis('off')
    
#    plt.tight_layout()
    return fig1, image_label_overlay

#%%
def affi_image(image,title='',close=True, rows=1, cols=2, color="gray"):
    """ function for display image
    return figure
    image : image input
    title : title
    close : for close previous windows.
    rows  : number of rows
    cols  : number of columns
    color : color
    """
    if close==True:
        plt.close('all')
    
    fig=plt.figure(title)
    fig.suptitle(title)
    axes1=fig.add_subplot(rows,cols,1)
    axes1.set_title('image orignale')
    axes1.imshow(image, cmap=color)
    axes1.axis("off")
    return fig    

#%%
def draw_ellipse_perso(image, classe):
    """ function for draw ellipse in image for display. draw blue ellipse.
    return image
    image  : image input.
    classe : list of region where draw ellipse.
    """
#    image=image*255
    image_color=np.array(image, dtype='uint8')

    for i in classe:
        r, c=i.centroid
        r_radius=i.major_axis_length
        c_radius=i.minor_axis_length
        orientation=i.orientation
        rr, cc=ellipse_perimeter(r=int(r),c=int(c),r_radius=int(c_radius/2),
                                 c_radius=int(r_radius/2),orientation=-orientation,
                                 shape=[image_color.shape[0],image_color.shape[1]])
        image_color[rr,cc,:]=[0,0,255]
        
    return image_color

#%%
def plot(matrice,titre):
    """ for plot a figure
    no return
    matrice : image
    titre   : title"""
    plt.figure()
    plt.title(titre)
    plt.imshow(matrice, cmap='gray')
    if True in matrice or False in matrice :
        pass
    else:
        plt.colorbar()
    plt.show()
    
#%%
def echelle(image,size,forme='circle', couleur='red',where=[10,10]):
    if forme=='circle':
        if len(image.shape)==2:
            cy,cx=circle(size+where[0],size+where[1],size,shape=image.shape)
            image_3D=np.zeros([image.shape[0],image.shape[1],3], dtype='uint8')
            image_3D[:,:,0]=image
            image_3D[:,:,1]=image
            image_3D[:,:,2]=image
            if couleur=='red':
                image_3D[cy,cx,:]=[255,0,0]
            elif couleur=='black':
                image_3D[cy,cx,:]=[0,0,0]
            elif couleur=='blue':
                image_3D[cy,cx,:]=[0,0,255]
            elif couleur=='green':
                image_3D[cy,cx,:]=[0,255,0]
            elif couleur=='white':
                image_3D[cy,cx,:]=[255,255,255]
            return image_3D
        
        elif len(image.shape)==3 and image.dtype=='uint8':
            cy,cx=circle(size+where[0],size+where[1],size,shape=image.shape)
            if couleur=='red':
                image[cy,cx,:]=[255,0,0]
            elif couleur=='black':
                image[cy,cx,:]=[0,0,0]
            elif couleur=='blue':
                image[cy,cx,:]=[0,0,255]
            elif couleur=='green':
                image[cy,cx,:]=[0,255,0]
            elif couleur=='white':
                image[cy,cx,:]=[255,255,255]
            return image
        else:
            #implementer la suite
            pass
    else:
        #implémenter d'autre forme
        pass