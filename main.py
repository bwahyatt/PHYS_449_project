## PRELIMINARY main.py script
## i.e. just a skeleton of what it could/should look like
## this script will be run after the raw/processed dataset generation

import torch
import numpy as np
import sys

sys.path.append('src')
from neural_net import Net

## this path can be made a variable/command line argument/json file parameter/etc later
## (or not)
processed_imgs_list = os.listdir("processed_images") 

## index for the cutoff of our training vs test data
## e.g. training data (file names) = processed_imgs[0:train_end_index], test data = processed_imgs[train_end_index:]
## this index can be made a "   "   "   "  "   "   "   " " " " " " " " " " " " " " 
train_end_index = 220 ## completely arbitrary placeholder
train_names = processed_imgs_list[0:train_end_index]
test_names = processed_imgs_list[train_end_index:]

## hyperparameters (get a json dict or something later.. )
## again, values are arbitrary placeholders
epochs = 10
learn_rate = 1e-3
hidden_nodes = 3 ## or whatever "third of the input size" the authors use
feature_size = 8 ## or 13 or 25 or whatever we want. But, it is something we have to pick by hand
                 ## to recreate paper, we will probably need to try all 3 author values?
    
## read in our labels
ids_and_labels = np.loadtxt('ids_and_labels.csv', skiprows=1) ## skip the column labels








