# Projet Cassioppee

This repo contains all the code for the project Cassioppee which is maintained by students from Telecom SudParis.
The goal of the project is to segment images.



## Clone the repository

First thing is to clone this repository: 
```bash
git clone https://github.com/rhajjar/projet_cassioppee.git
```
This will download a folder named projet_cassioppee.

## Create the Python virtual environment for the project




Go inside the folder projet_cassioppee :
```bash
cd projet_cassioppee
```

Create a virtual environment (this will create a folder).

Replace env_name below by the name of your environment. 

```bash
python3 -m venv env_name
```

To activate the environment run : 
```bash
source env_name/bin/activate
```

Once the environment has been activated, to deactivate it, run 
```bash
deactivate
```
If you want to know more about python virtual environment, check out the [documentation](https://docs.python.org/3/library/venv.html).

## Install the Python packages

First activate the  virtual environment.

Then run :
````bash
pip3 install -r requirements.txt
````
The environment is ready to be used with all the packages installed. 


## The Data structure

First, download some of the data on this [link](https://cutt.ly/ynYFQ64). (only 10 samples corresponding to 1 patient)

They are structured as follow :
- csv (parisitic data for each samples)
- inputs (the original data)
        - G (Green)[.tif and .bmp]
        - Labeled [.png]
        - B (blue)[.tif and .bmp]
        - R (red)[.tif and .bmp]
- outputs (the data outputed by the different algorithms)
        - save_segmentation (the outputs from the fonction segmentation)
        - fpm_84 (the outputs from the fonction FPM_84_save_cell)

- positions_des_leds (csv folders with the calculed by  hand position of the led)



## What have been done and what remains to be done

### Done

- Added relative path to the code Segmentation
- Tried to implement the same technic in the others codes, FPM_84_save_cell, iten_classification and phase_classification.
- Beginning transforming notebooks (.ipynb) into python file (.py) in order to call them for the interface code.
- Beginning to reorganize the data and clarify the documentation with the git and the README.
- Coded a graphic interface in order to show the results in a clear way and to centralise all the different code to do the Fourier ptychography.  

### To do 

- Continue to implement relative path to the codes.
- Then copy them into .py files 
- Improve the graphic interface.
- Improve the git by continuing to organize it.
- Comment the codes that already exist in order to apprehend them better and make them more comprehensible.
- Taking good habits if you code your own algorithms (comment, add readme, use coherent name of functions,use relative path, etc).
- And continue to make the git with the all folders the more homogeneous possible.  



## The Data structure

First, download some of the data on this [link](https://cutt.ly/ynYFQ64). (only 10 samples corresponding to 1 patient)

They are structured as follow :
- csv (parisitic data for each samples)
- inputs (the original data)
        - G (Green)[.tif and .bmp]
        - Labeled [.png]
        - B (blue)[.tif and .bmp]
        - R (red)[.tif and .bmp]
- outputs (the data outputed by the different algorithms)
        - save_segmentation (the outputs from the fonction segmentation)
        - fpm_84 (the outputs from the fonction FPM_84_save_cell)

- positions_des_leds (csv folders with the calculed by  hand position of the led)



## What have been done and what remains to be done

### Done

- Added relative path to the code Segmentation
- Tried to implement the same technic in the others codes, FPM_84_save_cell, iten_classification and phase_classification.
- Beginning transforming notebooks (.ipynb) into python file (.py) in order to call them for the interface code.
- Beginning to reorganize the data and clarify the documentation with the git and the README.
- Coded a graphic interface in order to show the results in a clear way and to centralise all the different code to do the Fourier ptychography.  

### To do 

- Continue to implement relative path to the codes.
- Then copy them into .py files 
- Improve the graphic interface.
- Improve the git by continuing to organize it.
- Comment the codes that already exist in order to apprehend them better and make them more comprehensible.
- Taking good habits if you code your own algorithms (comment, add readme, use coherent name of functions,use relative path, etc).
- And continue to make the git with the all folders the more homogeneous possible.  







