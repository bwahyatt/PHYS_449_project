## PRELIMINARY main.py script
## i.e. just a skeleton of what it could/should look like
## this script will be run after the raw/processed dataset generation

import torch
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sys
import os
import json
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

from src.neural_net import Net, GalaxiesDataset
import src.data_compression as dc  ## I think we need all of them 
from src.verbosity_printer import VerbosityPrinter
from src.remove_rogue_files import list_dir

def main():
    
    ## Maybe convert these to argparse later
    processed_images_dir = 'processed_images'
    hyperparams_path = './param/param.json'
    ids_and_labels_path = 'ids_and_labels.csv'
    system_verbosity = 2 # 2 = debug mode; 0 = performance report mode only; 1 = something in between
    
    ## this path can be made a variable/command line argument/json file parameter/etc later
    ## (or not)
    # processed_imgs_list = list_dir(processed_images_dir, '.DS_Store')

    ## Import hyperparameters from .json
    with open(hyperparams_path) as paramfile: ## could make the path a command line argument
        param = json.load(paramfile)

    ## added a couple others here/in the json
    batch = param['model']['batch']
    epochs = param['optim']['epochs']
    learn_rate = param['optim']['learn_rate']
    hidden_nodes = param['model']['hidden_nodes'] ## or whatever "third of the input size" the authors use
    feature_size = param['model']['feature_size'] ## or 13 or 25 or whatever we want. But, it is something we have to pick by hand
                                            ## to recreate paper, we will probably need to try all 3 author values?
    class_label_mapping = param['class_label_mapping']    
    num_class = len(class_label_mapping)
    
    ## Separate our testing and training data using a cutoff index
    ## index for the cutoff of our training vs test data
    ## e.g. training data (file names) = processed_imgs[0:train_end_index], test data = processed_imgs[train_end_index:]
    ## this index can be made a "   "   "   "  "   "   "   " " " " " " " " " " " " " " 
    
    train_dir = f'{processed_images_dir}/train'
    test_dir = f'{processed_images_dir}/test'
    
    # Create directories if they're missing, if images have already been partitioned, then unpartition them 
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    else:
        for fname in list_dir(train_dir, '.DS_Store'):
            shutil.move(f'{train_dir}/{fname}', f'{processed_images_dir}/{fname}')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    else:
        for fname in list_dir(test_dir, '.DS_Store'):
            shutil.move(f'{test_dir}/{fname}', f'{processed_images_dir}/{fname}')
    
    # Partition our images into a training dataset and a testing dataset
    train_end_index = param['model']['train_end_index'] ## moved this to param.json
    processed_imgs_list = [f for f in list_dir(processed_images_dir, '.DS_Store') if os.path.isfile(os.path.join(processed_images_dir, f))]
    for i, fname in enumerate(processed_imgs_list):
        if i <= train_end_index:
            shutil.move(f'{processed_images_dir}/{fname}', f'{train_dir}/{fname}')
        else:
            shutil.move(f'{processed_images_dir}/{fname}', f'{test_dir}/{fname}')
            
        
    # ## read in our labels
    # ids_and_labels = pd.read_csv(ids_and_labels_path) 
    
    # Initialize the verbosity printer
    vprinter = VerbosityPrinter(system_verbosity)
    
    # Compress and label the processed dataset
    class_col = 'Simple Classification'
    vprinter.vprint("Processing training data:", 1)
    train_dataset = GalaxiesDataset(processed_images_dir = train_dir, 
                                    ids_and_labels_path = ids_and_labels_path, 
                                    feature_size = feature_size, 
                                    class_label_mapping = class_label_mapping,
                                    class_col = class_col,
                                    vprinter = vprinter)
    vprinter.vprint("Processing testing data:", 1)
    test_dataset = GalaxiesDataset(processed_images_dir = test_dir, 
                                    ids_and_labels_path = ids_and_labels_path, 
                                    feature_size = feature_size, 
                                    class_label_mapping = class_label_mapping,
                                    class_col = class_col,
                                    vprinter = vprinter,                            
                                    train_dataset = train_dataset)
            
    ## NN stuff - cf. Workshop 2 / assignment 2
    model = Net(feature_size, hidden_nodes, num_class)
    optimizer = optim.SGD(model.parameters(), lr=learn_rate) 
    loss = nn.BCELoss()                        ## or, if we are only doing 2 classes, could use BCEloss?

    train_loss_list = []    ## plot this after training
    test_loss_list = []    ## plot this after training
    
    for epoch_count in tqdm(range(epochs), desc = "Training Epoch Count"):
        train_loss_val = model.train_model(galaxies_data = train_dataset, 
                                        loss_fn = loss, 
                                        optimizer = optimizer, 
                                        batch_size = batch,
                                        vprinter = vprinter)
        test_loss_val = model.test(galaxies_data = test_dataset,
                                    loss_fn = loss,
                                    vprinter = vprinter,
                                    batch_size = batch)
        train_loss_list.append(train_loss_val)
        test_loss_list.append(test_loss_val)
    
    ## Finally, test data
    ## can do it all as one big batch 
    test_loss = model.test(galaxies_data = test_dataset,
                            loss_fn = loss,
                            vprinter = vprinter,
                            batch_size = len(test_dataset),
                            show_accuracy = True
                            )    
    # test_dataset.save_eigengalaxies('sandbox/outputs')
    vprinter.vprint(f'TEST DATA LOSS: {test_loss}')
    
    ## training loss plot
    plt.plot(train_loss_list, label = 'Train Loss')
    plt.plot(test_loss_list, label = 'Test Loss')
    plt.title('Loss vs. Epochs')
    plt.ylabel('Loss Value')
    plt.xlabel('Epochs')
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which = 'major', color = 'darkgrey')
    plt.grid(True, which = 'minor', color = 'lightgrey')
    plt.xlim(1, epochs)
    try:
        plt.savefig('loss_plots.pdf')
    except PermissionError:
        vprinter.vprint("Error, unable to save loss plot file. Likely causes:\n- You have the pdf opened in another program\n- Not enough space in disk\n- You tried to save it in a folder with insufficient permissions",0)
    plt.show()
    0
    
if __name__ == '__main__':
    main()