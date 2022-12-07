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
import src.data_compression as dc 
from src.verbosity_printer import VerbosityPrinter
from src.remove_rogue_files import list_dir

def main():
    
    class_mode = 'multi'
    if class_mode == 'binary':
        processed_images_dir = 'grayscale_images'
        hyperparams_path = './param/binary_param.json'
        ids_and_labels_path = 'ids_and_labels.csv'
    elif class_mode == 'multi':
        processed_images_dir = 'grayscale_images'
        hyperparams_path = 'param/param.json'
        ids_and_labels_path = 'specific_ids_and_labels.csv'
    system_verbosity = 2 # 2 = debug mode; 0 = performance report mode only; 1 = something in between
    

    ## Import hyperparameters from .json
    with open(hyperparams_path) as paramfile:
        param = json.load(paramfile)

    batch = param['model']['batch']
    epochs = param['optim']['epochs']
    learn_rate = param['optim']['learn_rate']
    hidden_nodes = param['model']['hidden_nodes'] 
    feature_size = param['model']['feature_size'] 
    class_label_mapping = param['class_label_mapping']    
    num_class = len(class_label_mapping)
    if num_class == 2:
        num_class = 1
    
    ## Separate our testing and training data using a cutoff index
    ## index for the cutoff of our training vs test data
    ## e.g. training data (file names) = processed_imgs[0:train_end_index], test data = processed_imgs[train_end_index:]
    
    train_dir = f'{processed_images_dir}/train'
    test_dir = f'{processed_images_dir}/test'
    all_data_dir = f'{processed_images_dir}/all'
    
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
    shutil.copytree(processed_images_dir, all_data_dir, ignore = shutil.ignore_patterns('train*', 'test*', 'all*'), dirs_exist_ok = True)
    
    # Partition our images into a training dataset and a testing dataset
    train_end_index = param['model']['train_end_index']
    processed_imgs_list = [f for f in list_dir(processed_images_dir, '.DS_Store') if os.path.isfile(os.path.join(processed_images_dir, f))]
    for i, fname in enumerate(processed_imgs_list):
        if i <= train_end_index:
            shutil.move(f'{processed_images_dir}/{fname}', f'{train_dir}/{fname}')
        else:
            shutil.move(f'{processed_images_dir}/{fname}', f'{test_dir}/{fname}')
            
        
    
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
            
    ## Neural network stuff 
    model = Net(feature_size, hidden_nodes, num_class)
    optimizer = optim.SGD(model.parameters(), lr=learn_rate) 
    if num_class == 1:
        loss = nn.BCELoss()                       
    else: 
        loss = nn.CrossEntropyLoss()                  
        

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
    
    ## test data
    test_loss = model.test(galaxies_data = test_dataset,
                            loss_fn = loss,
                            vprinter = vprinter,
                            batch_size = len(test_dataset),
                            show_accuracy = True,
                            confusion_matrix_out_path = 'confusion_matrix.pdf'
                            )    
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