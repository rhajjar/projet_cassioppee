#%%
#importation 
import random, time, datetime, os, sys
sys.path.append(os.path.abspath('./'))
from fonction_densenet import create_densenet, creation_data, load_data#, load_densenet
from fonction_densenet import train_first, filter_part, train_continue, plot_train, SaveModel
from fonction_densenet import SavePartition, EvaluModel, VisuErreur, evalu_cascade, plot_matrix
from fonction_densenet import SaveHistory, loadHistory

from keras.optimizers   import Adam

from my_cascade import DataGenerator_empty#, DataGenerator_infected, get_labels_infected, get_labels_empty

import numpy as np

import pandas as pd


#%%
#paramters
debut=time.time()
datasets=['basic','different','ponderate','combo']
creation_cascade=True
input_shape=[84,84,3]
nb_classes = [3,2]

train = False
creation_dataset=False
name_data=["D:/Users/gourgue/Documents/Nicolas/BDD malaria/dataset_distrac/",
           "D:/Users/gourgue/Documents/Nicolas/BDD malaria/dataset_augmentation/"]
nb_epoch=[100,100]
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
name_model=[date_str+' première partie',date_str+' deuxième partie']

#%%
#if create modele
if creation_cascade:
    densenet_1=create_densenet(input_shape, nb_classes[0])
    densenet_1.name=date_str+' comparaison' 

else:
    densenet_1=load_densenet(name_model[0])
    history=loadHistory(folder_save+"history/"+name_model[0]+' history.mat')
    
#%%
#train creation
if creation_dataset:
    partition=creation_data(folders=name_data,color='G',dataset=datasets[3])
else:
    partition=load_data(name_data)

#shuffle
for key in partition.keys():
    random.shuffle(partition[key])
    
#generator creation
training_generator_1   = DataGenerator_empty(partition["train"], batch_size=batch_size, dim=input_shape[:2], 
                                   n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=shuffle, 
                                   empty='3')
validation_generator_1 = DataGenerator_empty(partition["valid"], batch_size=batch_size, dim=input_shape[:2], 
                                   n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=shuffle, 
                                   empty='3')
testing_generator_1    = DataGenerator_empty(partition["test" ], batch_size=batch_size, dim=input_shape[:2], 
                                       n_channels=input_shape[2], n_classes=nb_classes[0], shuffle=False, 
                                       empty='3')


#%%
#train
if train is True:
    train_first(densenet_1,training_generator_1, nb_epoch=nb_epoch[0], learning_rate=0.00001,
                                    momentum=0.9, Opt=Adam, loss='categorical_crossentropy',
                                    metrics=['accuracy'], validation_generator=validation_generator_1,
                                    verbose=verbose, weights=weights_1)
    

elif train == 'continue':    
    train_continue(densenet_1,training_generator_1, nb_epoch=100, learning_rate=0.00001, momentum=0.9, 
                   validation_generator=validation_generator_1, verbose=verbose, weights=weights_1)
    
    
else:
    pass

#%%
#visualisation
#%% visualisation
plot_train(densenet_1, title='first classifier')

#%%
#évaluation
EvaluModel(densenet_1, partition, testing_generator_1, mode='empty' )
VisuErreur(densenet_1, partition, testing_generator_1,folder_save=folder_save, mode='empty',
           SAVE=Save_error, color='G')


#%%
#Save
if Save_model:
    SaveModel(densenet_1, where=folder_save)
    SaveHistory(densenet_1, where=folder_save)

    
if Save_partition:
    SavePartition(densenet_1, partition, where=folder_save)
    
#%%
fin=time.time()
duration = fin-debut
hour = int(duration/3600)
minute = int((duration-60*hour)/60)
second = round(duration-3600*hour-60*minute,2)
print('duration: '+str(hour)+':'+str(minute)+":"+str(second))


    
    
