# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:55:33 2019

@author: gourgue
"""

#%%
#importation 
from fonction_densenet import create_densenet, creation_data, load_data
from fonction_densenet import filter_part, train_continue, plot_train, SaveModel
from fonction_densenet import SavePartition, EvaluModel, VisuErreur, evalu_cascade, plot_matrix

import random, time, sys, os
# sys.path.append(os.path.abspath('../'))

sys.path.append('/content/drive/My Drive/Stage/code_test_nicolas')

from function_model import train_first, LoadModel

from keras.optimizers   import Adam

from my_cascade import DataGenerator_empty#, DataGenerator_infected, get_labels_infected, get_labels_empty

import numpy as np

import pandas as pd


#%%
#paramters
debut=time.time()
datasets=['basic','different','ponderate','combo','cascade']

input_shape=[84,84,3]
nb_classes = [2,2]

train = True
creation_cascade=True
creation_dataset=True
name_data=["D:/Users/gourgue/Documents/Nicolas/BDD malaria/dataset_distrac_hand/",
           "D:/Users/gourgue/Documents/Nicolas/BDD malaria/dataset_augmentation_hand/"]
nb_epoch=[2,2]
batch_size=128
shuffle=True
verbose=True
weights_1=[1,1]
weights_2=[1,1]
Save_model=True
Save_partition=True
Save_error=True
folder_save="D:/Users/gourgue/Documents/Nicolas/SAVE/"
date=time.localtime()
date_str = str(date[0])+'-'+str(date[1])+'-'+str(date[2])
date_str_load ="2019-12-8"
name_model=[date_str_load+' première partie',date_str_load+' deuxième partie']
name_load_data =[folder_save+'ordi perso/ensemble/'+date_str_load+' première partie partition',
            folder_save+'ordi perso/ensemble/'+date_str_load+' deuxième partie partition']
augmentation=['distrac','infected','healthy']
#%%
#if create modele
if creation_cascade:
    densenet_1=create_densenet(input_shape, nb_classes[0])
    densenet_2=create_densenet(input_shape, nb_classes[1])
    densenet_1.name=date_str+' première partie' 
    densenet_2.name=date_str+' deuxième partie'
else:
    densenet_1=LoadModel(folder_save+"ordi perso/modele/model_convolution "+name_model[0])
    densenet_2=LoadModel(folder_save+"ordi perso/modele/model_convolution "+name_model[1])
    
#%%
#partition creation
if creation_dataset:
    partition=creation_data(folders=name_data,color='G',dataset=datasets[4], 
                            augmentation=augmentation, verbose=True)
    #shuffle
    for key in partition.keys():
        random.shuffle(partition[key])
else:
    partition=load_data(name_load_data[0])
    for part in partition.keys():
        if type(partition[part])==np.ndarray:
            partition[part]=list(partition[part])

#generator creation
training_generator_1   = DataGenerator_empty(partition["train"][:int(len(partition['train'])/2)], batch_size=batch_size, dim=input_shape[:2], 
                                   n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=shuffle, 
                                   empty='empty')
validation_generator_1 = DataGenerator_empty(partition["valid"], batch_size=batch_size, dim=input_shape[:2], 
                                   n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=shuffle, 
                                   empty='empty')
testing_generator_1    = DataGenerator_empty(partition["test" ], batch_size=batch_size, dim=input_shape[:2], 
                                       n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=False, 
                                       empty='empty')
#del healthy image
partition_filter=filter_part(partition)

training_generator_2   = DataGenerator_empty(partition_filter["train"], batch_size=batch_size, dim=input_shape[:2], 
                                   n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=shuffle, 
                                   empty='infected')
validation_generator_2 = DataGenerator_empty(partition_filter["valid"], batch_size=batch_size, dim=input_shape[:2], 
                                   n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=shuffle, 
                                   empty='infected')
testing_generator_2    = DataGenerator_empty(partition_filter["test" ], batch_size=batch_size, dim=input_shape[:2], 
                                       n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=False, 
                                       empty='infected')

#%%
#train
if train is True:
    train_first(densenet_1,training_generator_1, nb_epoch=nb_epoch[0], learning_rate=0.00001,
                                    momentum=0.9, Opt=Adam, loss='categorical_crossentropy',
                                    metrics=['accuracy'], validation_generator=validation_generator_1,
                                    verbose=verbose, weights=weights_1, save=True, Folder_save=folder_save)
    
    train_first(densenet_2,training_generator_2, nb_epoch=nb_epoch[1], learning_rate=0.00001,
                                    momentum=0.9, Opt=Adam, loss='categorical_crossentropy',
                                    metrics=['accuracy'], validation_generator=validation_generator_2,
                                    verbose=verbose, weights=weights_2, save=True, Folder_save=folder_save)

elif train == 'continue':    
    train_continue(densenet_1,training_generator_1, nb_epoch=100, learning_rate=0.00001, momentum=0.9, 
                   validation_generator=validation_generator_1, verbose=verbose, weights=weights_1)
    
    train_continue(densenet_2,training_generator_2, nb_epoch=100, learning_rate=0.00001, momentum=0.9, 
                   validation_generator=validation_generator_2, verbose=verbose, weights=weights_2)
    
else:
    opt= Adam(lr=0.00001)
    densenet_1.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    densenet_2.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#%%
#visualisation
#%% visualisation
plot_train(densenet_1, title='first classifier')

plot_train(densenet_2, title='second classifier')

#%%
#évaluation
EvaluModel(densenet_1, partition, testing_generator_1, mode='empty' )
VisuErreur(densenet_1, partition, testing_generator_1,folder_save=folder_save, mode='empty',
           SAVE=Save_error, color='G')

EvaluModel(densenet_2, partition_filter, testing_generator_2, mode='infected' )
VisuErreur(densenet_2, partition_filter, testing_generator_2,folder_save=folder_save, mode='empty',
           SAVE=Save_error, color='G')

#%%
#Save
if Save_model:
    SaveModel(densenet_1, where=folder_save)
    SaveModel(densenet_2, where=folder_save)
    
if Save_partition:
    SavePartition(densenet_1, partition, where=folder_save)
    SavePartition(densenet_2, partition_filter, where=folder_save)
    
#%%
#evaluate 
mat_conf = evalu_cascade(partition, testing_generator_1, densenet_1, densenet_2, nb_classes=nb_classes)

mat_bi = pd.DataFrame(np.zeros([2,2]))
mat_bi.set_axis(['healthy_t','infected_t'], axis=0, inplace=True)
mat_bi.set_axis(['healthy_p','infected_p'], axis=1, inplace=True)

mat_bi.at['healthy_t','healthy_p']=mat_conf.at["healthy_t",'healthy_p']+\
mat_conf.at["distrac_t",'distrac_p']+mat_conf.at["healthy_t",'distrac_p']+\
mat_conf.at["distrac_t",'healthy_p']

mat_bi.at['healthy_t','infected_p']=mat_conf.at["healthy_t","infected_p"]+mat_conf.at["distrac_t","infected_p"]
mat_bi.at['infected_t','healthy_p']=mat_conf.at["infected_t","healthy_p"]+mat_conf.at["infected_t","distrac_p"]
mat_bi.at["infected_t","infected_p"]=mat_conf.at["infected_t", "infected_p"]

plot_matrix(np.array(mat_conf), title='erreur detail')
plot_matrix(np.array(mat_bi), title="erreur final")

    
#%%
fin=time.time()
duration = fin-debut
hour = int(duration/3600)
minute = int((duration-60*hour)/60)
second = round(duration-3600*hour-60*minute,2)
print('duration: '+str(hour)+':'+str(minute)+":"+str(second))
    


    
    