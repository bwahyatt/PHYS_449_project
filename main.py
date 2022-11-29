## PRELIMINARY main.py script
## i.e. just a skeleton of what it could/should look like
## this script will be run after the raw/processed dataset generation

import torch
import numpy as np
import pandas as pd
import sys
import os
import json

from src.neural_net import Net
from src.data_compression import * ## I think we need all of them 

## this path can be made a variable/command line argument/json file parameter/etc later
## (or not)
processed_imgs_list = os.listdir("processed_images") 



## index for the cutoff of our training vs test data
## e.g. training data (file names) = processed_imgs[0:train_end_index], test data = processed_imgs[train_end_index:]
## this index can be made a "   "   "   "  "   "   "   " " " " " " " " " " " " " " 

train_end_index = 220 ## completely arbitrary placeholder
train_names = processed_imgs_list[0:train_end_index]
test_names = processed_imgs_list[train_end_index:]

## Import hyperparameters from .json
with open('param/param.json') as paramfile: ## could make the path a command line argument
    param = json.load(paramfile)

epochs = param['epochs']
learn_rate = param['learn_rate']
hidden_nodes = param['hidden_nodes'] ## or whatever "third of the input size" the authors use
feature_size = param['feature_size'] ## or 13 or 25 or whatever we want. But, it is something we have to pick by hand
                 ## to recreate paper, we will probably need to try all 3 author values?
    
## read in our labels
ids_and_labels = pd.read_csv('ids_and_labels.csv') 

## get a (global) mean image vector, and big cov. matrix, PCA matrix
## this will probably take a minute to run
proc_path = 'processed_images'
mean_vector = mean_image_vec(proc_path)                          ## mean flattened vector of whole dataset
print("mean vector acquired")

big_C_matrix = big_cov_matrix(mean_vector, proc_path)            ## big covariance matrix for whole dataset
print("big C acquired")

PCA_matrix = mat_of_thetas_to_pcs(big_C_matrix, feature_size)    ## matrix of big C's principle components
print("PCA matrix acquired")

## we can e.g. construct a numpy array with all of our dataset's feature vectors
    ## each row = feature vector
## note: if we vary the size of feature vectors, might need to adjust this somehow
## or, just run main.py every time we adjust the feature vector size (since it is a hyper parameter)

feature_array = np.zeros((len(ids_and_labels), feature_size), float) ## or int? see what pytorch inevitably complains about 

for k in range(train_end_index): 
    processed_fname = f'{proc_path}/{ids_and_labels.ID[k]}.jpg'
    current_flat_img = flattener(processed_fname)
    current_feature_vec = feature_extract(PCA_matrix, current_flat_img, mean_vector)
    
    feature_array[k,:] += current_feature_vec
    
## seems a bit weird, last couple look like they are all zeroes?    
print(feature_array[-1,:])


