# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:57:09 2019

@author: gourgue
"""

#%% importation
import os
import time
#%%
def creation_folder(path='', temps=True):
    """
    test si le dossier existe sinon il est crĂŠer.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    if temps:
        date =time.localtime()
        date_str = str(date[0])+'-'+str(date[1])+'-'+str(date[2])
        
        if not os.path.exists(path+date_str):
            os.makedirs(path+date_str)
        return path+date_str+'/'
    else:
        return path