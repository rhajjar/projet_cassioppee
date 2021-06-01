# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:16:00 2019

@author: gourgue

code pour gérer les données. 
create_train : à partir d'une liste de chemin d'image créer 3 ensembles train, valid et test
add_list : fonction pour ajouter deux liste en addtionnant les élemnts dedans et non 
    en concanténant les listes comme le fait naturellement python.
creation data : permet à partir d'une liste de chemin, du choix du canal, et du choix 
    du dataset d'extraire la liste des chemin et ensuite appel create train pour créer 3 
    ensemble train valid et test
load data : permet de charger les données.
save data : permet de sauvegarder les données.
filter_part : utile pour le cascadeur permet de filtrer une partie des exemples pour
    extraire un sous ensemble d'apprentissage. dans le cas présent le but est de viré
    toutes les images healthy en vue d'entrainté le second classifieur.
"""

#%%
#importation
import os, random, copy, sys

import scipy.io.matlab as mio

sys.path.append('/content/drive/My Drive/Stage/code_test_nicolas')


from function_verification import creation_folder

#from sklearn.utils   import class_weight

#import numpy as np

#%%
datasets=["basic","augmentation","ponderate","combo"]
#%% #create train, val, test
def create_train(image_infected, nb_split=0.8, nb_val=0.8):
    """
    permet de créer une liste contenant 3 liste qui sont les listes des objets pour 
    l'apprentissage, la validation et le test. cette fonction faire un tirage aléatoire
    dans liste garantissant un ensemble tiré au hazard mais que les images augmenter
    qui se suive lors de l'extraction resterons à la suite et donc dans le même ensemble.
    nb_split correspond à la proportion de l'ensemble train/val sur l'ensemble total
    nb_val correspond à la proportionde l'ensemble train sur l'ensemble train/val
    
    """
    point_depart = random.randint(0,len(image_infected)-1)
    if point_depart+len(image_infected)*nb_split*nb_val < len(image_infected) :
        #cas ou le point de départ est avant les 34 premier %
#        print("cas où point de départ est au début")
        image_train=image_infected[point_depart:point_depart+int(len(image_infected)*nb_split*nb_val)]
        
        if point_depart+len(image_infected)*nb_split< len(image_infected) :
            #cas ou le point de départ est avant les 20 premier %
#            print("cas où point de départ est tout au début")
            image_val=image_infected[point_depart+int(len(image_infected)*nb_split*nb_val):\
                                     point_depart+\
                                    int(len(image_infected)*nb_split)]
            
            if point_depart+len(image_infected)== len(image_infected):
                #cas ou le point de départ est 0
#                print("cas où point de départ est le début")
                image_test=image_infected[point_depart+int(len(image_infected)*nb_split):]
                
            else:
                image_test = image_infected[point_depart+int(len(image_infected)*nb_split):]
                nb_cell_case = len(image_test)
                nb_cell_rest = int(len(image_infected)*(1-nb_split))
                image_test = image_test + image_infected[:nb_cell_rest]
            
        else:
            image_val=image_infected[point_depart+int(len(image_infected)*nb_split*nb_val):]
            nb_cell_case = len(image_val)
            nb_cell_rest = int(len(image_infected)*(1-nb_split)*(1-nb_val))
            image_val = image_val + image_infected[:nb_cell_rest]
            
            image_test = image_infected[nb_cell_rest:int(len(image_infected)*(1-nb_split))]
        
    else:
        #cas ou le point de départ est au dela des 34 premier %
#        print("cas sans problème")
        image_train=image_infected[point_depart:]
        nb_cell_case=len(image_train)
        nb_cell_rest=int(len(image_infected)*nb_split*nb_val-nb_cell_case)
        image_train=image_train+image_infected[:nb_cell_rest]
        
        image_val  =image_infected[nb_cell_rest:nb_cell_rest+int(len(image_infected)*(nb_split)*\
                                                                 (1-nb_val))]
        image_test =image_infected[nb_cell_rest+int(len(image_infected)*(nb_split)*(1-nb_val)):\
                                nb_cell_rest+int(len(image_infected)*(nb_split)*(1-nb_val))+\
                                int(len(image_infected)*(1-nb_split))]
        
#    print(len(image_train),len( image_val), len(image_test))
    return [image_train,image_val,image_test]
    
def add_list_list(list1,list2):
    """
    permet de fusionner les élements dans une liste. 
    exemple :
        a=[1,2,3,4]
        b=a+a
        >>>b=[1,2,3,4,1,2,3,4]
        b=add_list_list(a,a)
        >>>b=[2,4,5,8]
        
    ici cette contion est utiliser pour concaténer les listes à l'intérieur de la liste 
    principal
    """
    if len(list1)==len(list2):
        list3=[]
        for i in range(len(list1)):
            list3=list3+[list1[i]+list2[i]]
    else:
        list3=[None]
    return list3         

#%% create data
def creation_data(folders=[None],color='G',dataset=datasets[3]):
    #extract path
    """
    la fonction permet de à partir d'une liste de chemin , du canal de couleur choisi,
    et du format de dataset selectionné génère une variable partition qui est un dictionnaire
    qui contient 3 liste avec les clés train valid et test pour les listes des chemins des
    images.
    
    amélioration extraction du chemin et le test de j'extrait telle donnée dans telle chemin
    doit être améliorer. soit le nom du chemin ne contient que le type de données à prendre.
    soit  trouvé une alternative. car le code ne fonctionne que pour le cascadeur
    """
    data={}
    for folder in folders:
#        print(folder)
        healthy=False
        distrac=False
        infected=False
        if folder is None:
            break
        #extraction path
        if "distrac" in folder:
            healthy=True
        if "augmentation" in folder:
            distrac=True
            infected=True
        patients=os.listdir(folder)
        for patient in patients:
            try :
                data[patient]
            except:
                data[patient]={}
                data[patient]["distrac"] =[]
                data[patient]["healthy"] =[]
                data[patient]["infected"]=[]
            
            if 'distrac' in folder:
                names=[folder+patient+'/'+color+'/RAW_0/']
            elif 'augmentation' in folder:
                name=folder+patient+'/'+color+'/RAW_0/'
                if patient in ["KPJ0","CAT01"]:
                    names=[name+'infected/',name+'distrac/']
                else:
                    names=[name+'distrac/']
                    
            for name in names:
                images_name=os.listdir(name)
                for image_name in images_name:
                    if "tophat" in image_name:
                        pass
                    else:
                        images=os.listdir(name+image_name)
                        for image in images:
                            if "distrac" in image and distrac:
                                data[patient]["distrac"].append(name+image_name+'/'+image)
                            if "healthy" in image and healthy:
                                data[patient]["healthy"].append(name+image_name+'/'+image)
                            elif "infected" in image and infected:
                                data[patient]["infected"].append(name+image_name+'/'+image)
            else:
                pass
                            
    #repartition of data per patient
    image_infected=[]
    for i, image in enumerate(data['KPJ0']['infected']):
        if i%4==3:
            image_infected.append(data['CAT01']['infected'][int(i/4)])
        image_infected.append(image)
    if dataset==datasets[1]:
        image_healthy=data["CAT01"]["healthy"]
        nb_cells=int((len(image_infected)-len(image_healthy))/3)
        
        for patient in ['KPJ0','DA','LE']:
            cells=random.sample(data[patient]['healthy'], nb_cells )
            image_healthy=image_healthy+cells
    elif dataset==datasets[3]:
        image_healthy=[]
        for patient in patients:
            image_healthy=image_healthy+data[patient]["healthy"]
    
    image_distrac=[]
    saute_DA=[2,8]
    saute_LE=[4,6]
    for i, image in enumerate(data['KPJ0']['distrac']):
        image_distrac.append(image)
        if i%10 in saute_DA:
            pass
        else:
            image_distrac.append(data["DA"]["distrac"][int(i*len(data["DA"]["distrac"])/\
                                 len(data['KPJ0']['distrac']))])
            if i%10 in saute_LE:
                pass
            else:
                image_distrac.append(data["LE"]["distrac"][int(i*len(data["LE"]["distrac"])/\
                                     len(data['KPJ0']['distrac']))])
                image_distrac.append(data["CAT01"]["distrac"][int(i*len(data["CAT01"]["distrac"])/\
                                     len(data['KPJ0']['distrac']))])
                
    #creation train val and test
    partition={'train':[],'valid':[],'test':[]}
#    print(type(partition))
    if len(image_infected)==0:
        return 'partition_inf'  
#    print(len(image_infected))      
    partition["train"],partition["valid"], partition["test"] = create_train(image_infected)
#    print(type(partition))
    if len(image_distrac)==0:
        return 'partition_dis'
    partition["train"],partition["valid"], partition["test"] = add_list_list([partition["train"],
              partition["valid"], partition["test"]],create_train(image_distrac))
#    print(type(partition))
    if len(image_healthy)==0:
        return  'partition_heal'
    partition["train"],partition["valid"], partition["test"] = add_list_list([partition["train"],
              partition["valid"], partition["test"]],create_train(image_healthy))
#    print(type(partition))
    
    return partition

#%% Load/Save
def load_data(name_data):
    return mio.loadmat(name_data)
    
def save_data(name_data, data):
    path, name =os.path.split(name_data)
    path=creation_folder(path, temps=False)
    mio.savemat(name_data,data)
    print("data save in: ",name_data)
#%%
def filter_part(partition):
    """
    fonction uniquement pour la cascade.
    permet de viré les cellules saines et garde uniquement distrac et infected pour passer
    à la deuxième cascade.
    """
    partition_1=copy.deepcopy(partition)
    for key in partition_1.keys():
        if '__' in key:
            pass
        else:
            partition_1[key]=list(partition_1[key])
            for i in range(len(partition_1[key])-1,-1,-1):
                if 'healthy' in partition_1[key][i]:
                    partition_1[key].pop(i)
    
    return partition_1
    