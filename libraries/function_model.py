# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:41:58 2019

@author: gourgue
"""
"""
Création de fonction pour automatisé la génération et l'entrainement de modèle 
sur la malaria. il reste cependant des fonctions a généralisé car elles ont été créer
à partir du code du classifieur en cascade et donc en sont pas toujours compatible
avec une apprentissage simple classique.

"""
#%%
#importation
from keras.layers       import Dense, GlobalAveragePooling2D, Dropout
from keras.models       import Model, load_model
from keras.callbacks    import History, ModelCheckpoint, CSVLogger
from keras.applications import DenseNet201,ResNet101,ResNet152
from keras              import backend, layers, models, utils
from keras.optimizers   import Adam

import numpy as np
import pandas as pd

import scipy.io.matlab as mio

from sklearn.metrics import confusion_matrix

import sys, os, copy, time, datetime
date=time.localtime()
date_str=str(date[0])+'-'+'%02d'%date[1]+'-'+'%02d'%date[2]
folder_code="D:/Users/gourgue/Documents/Nicolas/codes"
sys.path.append(os.path.join(folder_code,date_str+' compteur'))

sys.path.append('/content/drive/My Drive/Stage/code_test_nicolas')

from my_cascade import get_labels_empty, DataGenerator_empty, get_labels_infected
from function_affiche import plot_matrix
#%%

archis=['resnet_101','densenet_201','resnet_152']

#%% creation 
def create_model(archi,input_shape=[84,84,3], nb_classes=2, pre_train=None,
                 activation='softmax'):
    """
    cette fonction créer une réseau resnet 101 ou densenet 101 à partir de la 
    bibliothèque keras. Le réseau est tronqué au niveau de la dernière couche 
    de convolution. On ajoute donc un global average pooling pour poser les 
    tenseurs dans un vecteurs on ajout un drop out et une couche de sortie. 
    (il aurait pu être intéressant d'ajouté une couche totalement connecté 
    supplémentaire.)
    
    paramètres:
        archi : 0 -> ResNet_101
                1 -> Densenet 201
                2 -> Resnet_152
        input_shape : tenseur d'entrée par défaut (X,X,3) X,X étant la taille de l'image à rentrée.
        nb_classe : nombre de classe
        pre_train : pre entraintemant ou pas. par défaut non mais il est possible d'utlisé le réseau
                    pré entrainer sur image net.
    """
    if archi==archis[0]:
         model = ResNet101(include_top=False, weights=pre_train,
                input_tensor=None, input_shape=input_shape, classes=nb_classes, backend=backend, 
                layers=layers, models=models, utils=utils)
    elif archi==archis[1]:
        model = DenseNet201(include_top= False, weights=pre_train, input_tensor=None, 
                                                  input_shape=input_shape, pooling=None, 
                                                  classes = nb_classes)
    elif archi==archis[2]:
        model = ResNet152(include_top=False, weights=pre_train,
                input_tensor=None, input_shape=input_shape, classes=nb_classes, backend=backend, 
                layers=layers, models=models, utils=utils)        
    
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predi = Dense(nb_classes, activation=activation)(x)
    return Model( inputs = model.input, outputs = predi)

#%%load
def LoadModel(name):
    """
    cette fonction a pour but de charger un modèle complet. c'est a dire les poids
    l'architecture, l'entrainement précédent et tout les paramètres propre au modèle.
    """
    model = load_model(name+'.h5')
    path, model_name = os.path.split(name)
    path, model_folder =os.path.split(path)
    setattr(model, 'history',History )
    #recupe params
    params=pd.read_csv(os.path.join(path,'history',model.name+' params.csv'))
    columns=params.columns
    dico={}
    for col in columns:
        if 'Unnamed'in col:
            pass
        else:
            param = params[col]
            if len(param)>1:
                value_p=None
                dico[col]=[]
                for value in param:
                    if value is value_p or value == value_p:
                        pass
                    else:
                        dico[col].append(value)
                    value_p=value
                if len(dico[col])==1:
                    dico[col]=dico[col][0]
    setattr(model.history, 'params',dico )
    #recupe history
    history=mio.loadmat(os.path.join(path,'history',model.name+' history.mat'))
    History_metrics={}
    for key in history.keys():
        if key in model.history.params["metrics"]:
            History_metrics[key]=history[key][0]
    setattr(model.history, 'history',History_metrics )
    #creaction de epoch
    setattr(model.history, 'epochs',list(range(model.history.params['epochs'])) )
        #problème a réglé
    print("modèle chargé")
    return model

#%% train

def train_first(model,training_generator, nb_epoch=100, learning_rate=0.00001,
                momentum=0.9, Opt=Adam, loss='categorical_crossentropy',
                metrics=['accuracy'], validation_generator=None, verbose=False, 
                weights=None, save=False, Folder_save=None):
    """
    cette fonction entraine un modèle avec possibilité se sauvegarder régulièrement 
    le modèle et les valeurs d'erreurs.
    """
    print("begin training")
    time_begin=time.time()
    opt= Opt(learning_rate, momentum)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    
    if save and not Folder_save is None: 
        if not os.path.exists(Folder_save+'checkpointer/'):
            os.mkdir(Folder_save+'checkpointer/')
        if not os.path.exists(Folder_save+'history/'):
            os.mkdir(Folder_save+'history/')
        
        archi=model.name[:model.name.find(' ')]    
        trainpath=Folder_save+'history/'+archi+' train.log'
        csv_logger = CSVLogger(trainpath)
        modelpath=Folder_save+'checkpointer/'+archi+' epoch={epoch:03d}-val_acc={val_acc:.2f}.hdf5'
        
        checkpointer = ModelCheckpoint(filepath=modelpath, monitor="val_acc", verbose=verbose, 
                                       save_best_only=False, save_weights_only=False, period=10)
        
        model.fit_generator(generator=training_generator, validation_data=\
                                         validation_generator, epochs=nb_epoch, verbose=verbose,
                                         class_weight=weights,callbacks=[checkpointer,csv_logger])
    
    else:
        model.fit_generator(generator=training_generator, validation_data=\
                                         validation_generator, epochs=nb_epoch, verbose=verbose,
                                         class_weight=weights)
    time_end=time.time()
    duration=time_end-time_begin
    delta=datetime.timedelta(0,duration)
    print("duration of training:",str(delta))
    return None

## a ajuster si l'entrainement écrase ou pas
def train_continue(model,training_generator, history_pred, nb_epoch=100, 
                   learning_rate=0.00001,momentum=0.9, validation_generator=None, 
                   verbose=False, weights=None):
    """
    cette méthode a pour but de continué un entrainement et donc d'inserer la suite
    de l'apprentissage dans le modèle.
    """
    print("continue training")
    time_begin=time.time()
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
    
    time_end=time.time()
    duration=time_end-time_begin
    delta=datetime.timedelta(0,duration)
    print("duration of training:",str(delta))
    return None

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
#%% evalutation
def EvaluModel(model, partition, testing_generator, mode='empty',verbose=False):
    """
    cette méthode evalue les performances d'un modèle. la première approche est
    de mesurer l'erreur et l'accuracy totale ensuite elle complète les informa-
    tion avec une matrice de confusion.
    """
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
    if verbose:
        plot_matrix(matrix)
        
def evalu_cascade(partition, testing_generator, model1, model2, nb_classes=[2,2]):
    """
    Le but ici est d'évalué les performances d'un classifieur en cascade. Pour 
    ce faire on prend un ensemble on test le premier r
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
    Y_true_1= get_labels_infected(partition['test'])[:len(exit_1)]
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
    Y_true_2 = get_labels_infected(input_2)[:len(exit_2)]
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
#%% save
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
