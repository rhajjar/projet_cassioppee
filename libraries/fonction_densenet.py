# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:06:10 2019

@author: gourgue
"""

#%%
#importation
from keras.applications import DenseNet201
from keras.layers       import Dense, GlobalAveragePooling2D, Dropout
from keras.models       import Model, load_model
from keras.models       import model_from_yaml
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks    import History

import os, copy, random, sys
sys.path.append(os.path.abspath("../"))


from my_cascade import get_labels_empty, get_labels, DataGenerator_empty, get_labels_infected

#import time


import scipy.io.matlab as mio

from sklearn.metrics import confusion_matrix
#from sklearn.utils   import class_weight

import numpy as np

#from my_classes import DataGenerator, get_labels

import matplotlib.pyplot as plt

from keras.optimizers   import Adam

from skimage.io import imread, imsave

import pandas as pd

#%%
#parameters
datasets=['basic','different','ponderate','combo','cascade']
folders=["D:/Users/gourgue/Documents/Nicolas/BDD malaria/dataset_distrac/",\
         "D:/Users/gourgue/Documents/Nicolas/BDD malaria/dataset_augmentation/"]
#patients=['CAT01','KPJ0','DA','LE']
#%%
def create_densenet(input_shape=[84,84,3], nb_classes=2, pre_train=None):
    densenet = DenseNet201(include_top= False, weights=pre_train, input_tensor=None, 
                                                  input_shape=input_shape, pooling=None, classes = nb_classes)
    
    x = densenet.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predi = Dense(nb_classes, activation='softmax')(x)
    return Model( inputs = densenet.input, outputs = predi)



def create_train(image_infected, nb_split=0.8, nb_val=0.8):
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
    if len(list1)==len(list2):
        list3=[]
        for i in range(len(list1)):
            list3=list3+[list1[i]+list2[i]]
    else:
        list3=[None]
    return list3         


def creation_data(folders=[None],color='G',dataset=datasets[3],
                  augmentation=['infected','distrac'], 
                  verbose=False):
    
    #extract path
    data={}
    for folder in folders:
#        print(folder)
        
        if folder is None:
            break
        #extraction path
        if "augmentation" in folder:
            if 'infected' in augmentation:
                 infected=True
            elif not 'infected' in augmentation:
                infected=False
            if 'distrac' in augmentation:
                distrac=True
            elif not 'distrac' in augmentation:
                distrac = False
            if 'healthy' in augmentation:
                healthy=True
            elif not "healthy" in augmentation:
                healthy=False
            
        elif not "augmentation" in folder:
            if not 'infected' in augmentation:
                 infected=True
            elif 'infected' in augmentation:
                infected=False
            if not 'distrac' in augmentation:
                distrac=True
            elif 'distrac' in augmentation:
                distrac = False
            if not 'healthy' in augmentation:
                healthy=True
            elif "healthy" in augmentation:
                healthy=False
                
        patients=os.listdir(folder)
        for patient in patients:
            try :
                data[patient]
            except:
                data[patient]={}
                data[patient]["distrac"] =[]
                data[patient]["healthy"] =[]
                data[patient]["infected"]=[]
            
            if 'augmentation' in folder:
                names=[]
                for augment in augmentation:
                    names.append(folder+patient+'/'+color+\
                                 '/RAW_0/'+augment+'/')
#                names=[folder+patient+'/'+color+'/RAW_0/']
            elif not 'augmentation' in folder:
                names=[folder+patient+'/'+color+'/RAW_0/']
#                name=folder+patient+'/'+color+'/RAW_0/'
#                if patient in ["KPJ0","CAT01"]:
#                    names=[name+'infected/',name+'distrac/']
#                else:
#                    names=[name+'distrac/']
            
                    
            for name in names:
                if os.path.exists(name):
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
    if verbose:
        print("Patient , Distrac , Healthy , Infected")
        print("Cat01:",len(data["CAT01"]["distrac"]),len(data["CAT01"]["healthy"]),
              len(data["CAT01"]["infected"])) 
        print("KPJ0:",len(data["KPJ0"]["distrac"]),len(data["KPJ0"]["healthy"]),
              len(data["KPJ0"]["infected"]))
        print("DA:",len(data["DA"]["distrac"]), len(data["DA"]["healthy"]), 
              len(data["DA"]["infected"]))
        print("LE :", len(data["LE"]["distrac"]), len(data["LE"]["healthy"]),
              len(data["LE"]["infected"]),"\n")
    #repartition of data per patient
    if dataset in datasets[:4]:
        
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
                    
        
    elif dataset==datasets[4]:
        
        image_infected=[]
        compt_inf_CAT01=0
        compt_inf_KPJ0=0
        for i in range(len(data['KPJ0']['infected'])*2):
            if i%2==0:
                if compt_inf_CAT01<len(data['CAT01']['infected']):
                    image_infected.append(data['CAT01']['infected'][compt_inf_CAT01])
                    compt_inf_CAT01+=1
            elif i%2==1:
                if compt_inf_KPJ0< len(data['KPJ0']['infected']):
                    image_infected.append(data['KPJ0']['infected'][compt_inf_KPJ0])
                    compt_inf_KPJ0+=1
        
            
        image_distrac=[]
        compt_dis_CAT01=0
        compt_dis_KPJ0=0
        compt_dis_DA=0
        compt_dis_LE=0
        for i in range(int(len(data['DA']['distrac'])/4*11)):
            if i%11==0:
                #DA1
                if len(data["DA"]["distrac"])-1>=compt_dis_DA:
                    image_distrac.append(data["DA"]["distrac"][compt_dis_DA])
                    compt_dis_DA+=1
            elif i%11==1:
                #CAT011
                if len(data["CAT01"]["distrac"])-1>=compt_dis_CAT01:
                    image_distrac.append(data["CAT01"]["distrac"][compt_dis_CAT01])
                    compt_dis_CAT01+=1
            elif i%11==2:
                #KPJ01
                if len(data["KPJ0"]["distrac"])-1>=compt_dis_KPJ0:
                    image_distrac.append(data["KPJ0"]["distrac"][compt_dis_KPJ0])
                    compt_dis_KPJ0+=1
            elif i%11==3:
                #LE1
                if len(data["LE"]["distrac"])-1>=compt_dis_LE:
                    image_distrac.append(data["CAT01"]["distrac"][compt_dis_LE])
                    compt_dis_LE+=1
            elif i%11==4:
                #DA2
                if len(data["DA"]["distrac"])-1>=compt_dis_DA:
                    image_distrac.append(data["DA"]["distrac"][compt_dis_DA])
                    compt_dis_DA+=1
            elif i%11==5:
                #DA3
                if len(data["DA"]["distrac"])-1>=compt_dis_DA:
                    image_distrac.append(data["DA"]["distrac"][compt_dis_DA])
                    compt_dis_DA+=1
            elif i%11==6:
                #CAT012
                if len(data["CAT01"]["distrac"])-1>=compt_dis_CAT01:
                    image_distrac.append(data["CAT01"]["distrac"][compt_dis_CAT01])
                    compt_dis_CAT01+=1
            elif i%11==7:
                #KPJ02
                if len(data["KPJ0"]["distrac"])-1>=compt_dis_KPJ0:
                    image_distrac.append(data["KPJ0"]["distrac"][compt_dis_KPJ0])
                    compt_dis_KPJ0+=1
            elif i%11==8:
                #LE2
                if len(data["LE"]["distrac"])-1>=compt_dis_LE:
                    image_distrac.append(data["CAT01"]["distrac"][compt_dis_LE])
                    compt_dis_LE+=1
            elif i%11==9:
                #DA4
                if len(data["DA"]["distrac"])-1>=compt_dis_DA:
                    image_distrac.append(data["DA"]["distrac"][compt_dis_DA])
                    compt_dis_DA+=1
            elif i%11==10:
                #CAT013
                if len(data["CAT01"]["distrac"])-1>=compt_dis_CAT01:
                    image_distrac.append(data["CAT01"]["distrac"][compt_dis_CAT01])
                    compt_dis_CAT01+=1            
                
            
        image_healthy=[]
        for i in range(len(data["CAT01"]["healthy"])):
            image_healthy.append(data["CAT01"]["healthy"][i])
            if i<len(data['KPJ0']['healthy']):
                image_healthy.append(data["KPJ0"]["healthy"][i])
            if i<len(data['DA']['healthy']):
                image_healthy.append(data["DA"]["healthy"][i])
            if i<len(data["LE"]['healthy']):
                image_healthy.append(data["LE"]["healthy"][i])
    
    if verbose =='all':
        print("Distrac")
        print("compt CAT01:",compt_dis_CAT01)
        print("compt KPJ0:",compt_dis_KPJ0)
        print("compt DA:",compt_dis_DA)
        print("compt LE:",compt_dis_LE,"\n")
        print("infected")
        print("compt CAT01 :",compt_inf_CAT01)
        print("compt KPJ0 :", compt_inf_KPJ0,"\n")
    if verbose:
        print("distrac:", len(image_distrac))
        print("healthy:",len(image_healthy))
        print("infected:", len(image_infected))            
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

def load_data(name_data):
    return mio.loadmat(name_data)
    
def save_data(name_data, data):
    mio.savemat(name_data,data)
    print("data save in: ",name_data)

def train_first(model,training_generator, nb_epoch=100, learning_rate=0.00001,momentum=0.9, Opt=Adam, 
                loss='categorical_crossentropy',metrics=['accuracy'], validation_generator=None,
                verbose=False, weights=None):
    print("begin training")
    
    opt= Opt(learning_rate, momentum)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    model.fit_generator(generator=training_generator, validation_data=validation_generator, 
                                   epochs=nb_epoch, verbose=verbose,class_weight=weights)
    return None

def train_continue(model,training_generator, history_pred, nb_epoch=100, learning_rate=0.00001,momentum=0.9,
                validation_generator=None, verbose=False, weights=None):
    print("continue training")
    try :
        loss=model.loss
    except:
        loss='categorical_crossentropy'
    try :
        Opt=model.optimizer
    except:
        Opt=Adam
    try :
        metrics=model.metrics
    except:
        metrics=['accuracy']
    
    opt= Opt(learning_rate, momentum)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    model.fit_generator(generator=training_generator, validation_data=validation_generator, 
                                   epochs=nb_epoch, verbose=verbose,class_weight=weights)
    model.history.params= addition_dico(history_pred.params, model.history.parmas)
    model.history.history= addition_dico(history_pred.history, model.history.history)
    model.history.epoch=list(range(model.history.params['epochs']))
    
    return None

def filter_part(partition):
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

def addition_dico(dico1, dico2):
    dico3=copy.deepcopy(dico1)
    for key in dico2.keys():
        if key in dico3.keys():
            if key=='metrics':
                pass
            dico3[key]=dico3[key]+dico2[key]
        else:
            dico3[key]=dico2[key]
    return dico3

def plot_train(model, title='train'):
    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(model.history.history["acc"])
    plt.plot(model.history.history["val_acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","validation"],loc='upper left')
    plt.show()
    
    plt.subplot(1,2,2)
    plt.plot(model.history.history['loss'])
    plt.plot(model.history.history['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
    

def SaveModel(model, where=''):
    #test folder
    if not os.path.exists(where+'modele/'):
        os.mkdir(where+'modele/')
        
    #save model and history
    
    #save history.params
    params=pd.DataFrame(model.history.params)
    params.to_csv(where+'history/'+model.name+' params.csv')
    #save hsitory.history
    Histoire=model.history.history
    mio.savemat(where+'history/'+model.name+' history.mat',Histoire)
    #save model
    model.save(where+'modele/'+'model_convolution '+model.name+'.h5')

  
def LoadModel(name):
    model = load_model(name)
    path, model_name = os.path.split(name)
    path, model_folder =os.path.split(path)
    setattr(model, 'history',History )
    #recupe history
    history=mio.loadmat(os.path.join(path,'history',model.name+' history.mat'))
    setattr(model.history, 'history',history )
    #recupe params
    params=pd.read_csv(os.path.join(path,'history',model.name+' params.csv'))
    setattr(model.history, 'params',params )
    #creaction de epoch
    setattr(model.history, 'history',list(range(model.history.params['epochs'])) )
        #problème a réglé
    return model
    
def SavePartition(model, partition, where=''):
    if not os.path.exists(where+'ensemble/'):
        os.mkdir(where+'ensemble/')
    mio.savemat(where+'ensemble/'+model.name+' partition.mat',partition) 
    
    
def EvaluModel(model, partition, testing_generator, mode='empty'):
    scores = model.evaluate_generator(generator=testing_generator)

    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
    Y_pred = model.predict_generator(generator=testing_generator)
    Y_pred = np.argmax(Y_pred, axis=1)
    if mode=='empty':
        Y_test = get_labels_empty(partition['test'])[:len(Y_pred)]
    elif mode=='infected':
         Y_test = get_labels_infected(partition['test'])[:len(Y_pred)]
         
    Y_test = Y_test.reshape([len(Y_test),])
    Y_test = np.array(Y_test, dtype='uint64')
    
    
    matrix = confusion_matrix(Y_test, Y_pred)
    plot_matrix(matrix)
#%%
def VisuErreur(model, partition, testing_generator,folder_save='', mode='empty',SAVE=True, color='G'):
    #%%visualisation error
    
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
            
            
def evalu_cascade(partition, testing_generator, model1, model2, nb_classes=[2,2]):
    """
    indice healthy : prediction healthy
    indice full    : prediction candidate
    goof empty     : cell healthy classed as healthy
    tn_1           : number of cell true negative first classifier
    accept_empty   : cell distrac classed as healthy
    accept_fn_1    : number of cell false negative but real classe is distract for fisrt classifier
    error_empty    : cell infected classed as healthy
    error_fn_1     : number of cell false negative and real classe is infected for first classifier
    
    indice ditrac   : prediction distrac
    indice infected : prediction infected
    accept distrac  : cell healthy classed as distrac
    accept_err_2    : number of cell healthy classed as distrac
    good distrac    : cell distrac classed as distrac
    tn_2            : number of cell distrac classed as distrac
    error distrac   : cell infected classed as distrac
    fn_2            : number of cell infected classed as distrac
    
    error infected heal : cell healthy classed as infected
    fp_12               : number of cell helathy classed as infected
    error infected dist : cell distrac classed as infected
    fp_2                : number of cell distrac classed as infected
    good inected        : cell infected classed as infected
    tp_2                : number of cell infected classed as infected
    """
    
    mat_conf=np.zeros([3,3])
    mat_conf=pd.DataFrame(mat_conf)
    mat_conf.set_axis(['healthy_t','distrac_t','infected_t'], axis=0, inplace=True)
    mat_conf.set_axis(['healthy_p','distrac_p','infected_p'], axis=1, inplace=True)
    
    #premier passage
    exit_1=model1.predict_generator(generator=testing_generator)
    exit_1D = np.argmax(exit_1, axis=1)
    
#    #exit empty or full
#    Y_true_1 = get_labels_empty(partition['test'])[:len(exit_1)]
#    Y_true_1 = np.array(Y_true_1, dtype='uint64')
#    Y_true_1 = Y_true_1.reshape([len(Y_true_1),])    
    
    #exit infected or distract or healthy
    Y_true_1= get_labels(partition['test'])[:len(exit_1)]
    Y_true_1= np.array(Y_true_1, dtype='int8')
    Y_true_1= Y_true_1.reshape([len(Y_true_1),])
        
    #empty
    indice_empty = np.where(exit_1D==0)[0]
    #True negatif
    good_empty = np.where(Y_true_1[indice_empty]==0)[0]
    tn_1= len(good_empty)
    mat_conf.at['healthy_t','healthy_p']=tn_1
    #false negative : image full distrac predit empt
    accept_empty = np.where(Y_true_1[indice_empty]==1)[0]
    accept_fn_1= len(accept_empty)
    mat_conf.at['distrac_t','healthy_p']=accept_fn_1
    #false negative : image full infected predit empt
    error_empty = np.where(Y_true_1[indice_empty]==2)[0]
    error_fn_1= len(error_empty)
    mat_conf.at['infected_t','healthy_p']=error_fn_1

    #full
    indice_full = np.where(exit_1D==1)[0]
#    #false positive : image empty predit full
#    error_full = np.where(Y_true_1[indice_full]==-1)[0]
#    error_fp_1 = len(error_full)
    
    
    
    #Params
    batch_size=128
    input_shape=[84,84,3]
    #creation second ensemble
    input_2 = np.array(partition['test'])[indice_full]
    input_2 = list(input_2)
    testing_generator_2    = DataGenerator_empty(input_2, batch_size=batch_size, 
                                                 dim=input_shape[:2], n_channels=input_shape[2], 
                                                 n_classes=nb_classes[0], shuffle=False, empty='infected')
    #second passage
    exit_2  = model2.predict_generator(generator=testing_generator_2)
    exit_2D = np.argmax(exit_2, axis=1)
    
    #exit infected or distract or healthy
    Y_true_2 = get_labels(input_2)[:len(exit_2)]
    Y_true_2 = np.array(Y_true_2, dtype='int8')
    Y_true_2 = Y_true_2.reshape([len(Y_true_2),]) 
    
    #Distrac
    indice_distrac = np.where(exit_2D==0)[0]
    
    #empty : image healthy classed as ditrac
    accept_heal = np.where(Y_true_2[indice_distrac]==0)[0]
    accept_err_2= len(accept_heal)
    mat_conf.at['healthy_t','distrac_p']=accept_err_2
    
    #true negative : image distrac classed as distrac
    good_dist = np.where(Y_true_2[indice_distrac]==1)[0]
    tn_2  = len(good_dist)
    mat_conf.at['distrac_t','distrac_p']=tn_2
    
    #false negative
    error_distrac = np.where(Y_true_2[indice_distrac]==2)[0]
    fn_2          = len(error_distrac)
    mat_conf.at['infected_t','distrac_p']=fn_2
    
    #infected
    indice_infected = np.where(exit_2D==1)[0]
    
    #empty : image healthy classed as infected
    error_infected_heal = np.where(Y_true_2[indice_infected]==0)[0]
    fp_12               = len(error_infected_heal)
    mat_conf.at['healthy_t','infected_p']=fp_12
    
    #false positive : image distrac classed as infected
    error_infected_dist = np.where(Y_true_2[indice_infected]==1)[0] 
    fp_2                = len(error_infected_dist)
    mat_conf.at['distrac_t','infected_p']=fp_2
    
    #true positive : image infected classed as infected
    good_inected = np.where(Y_true_2[indice_infected]==2) [0]
    tp_2         = len(good_inected)
    mat_conf.at['infected_t','infected_p']=tp_2
    

    
    return mat_conf
    
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
    
def SaveHistory(model, where=''):
    if not os.path.exists(where+'history/'):
        os.mkdir(where+'history/')
    mio.savemat(where+'history/'+model.name+' history.mat',model.history) 
    
def loadHistory(name):
    return mio.loadmat(name)
    



    