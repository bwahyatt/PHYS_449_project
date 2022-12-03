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

def main():
    
    ## Maybe convert these to argparse later
    processed_images_dir = 'processed_images'
    hyperparams_path = 'param/param.json'
    ids_and_labels_path = 'ids_and_labels.csv'
    system_verbosity = 2 # 2 = debug mode; 0 = performance report mode only; 1 = something in between
    
    ## this path can be made a variable/command line argument/json file parameter/etc later
    ## (or not)
    # processed_imgs_list = os.listdir(processed_images_dir) 
    processed_imgs_list = [f for f in os.listdir(processed_images_dir) if os.path.isfile(os.path.join(processed_images_dir, f))]

    ## Import hyperparameters from .json
    with open(hyperparams_path) as paramfile: ## could make the path a command line argument
        param = json.load(paramfile)

    ## added a couple others here/in the json
    num_class = param['num_class']
    batch = param['batch']
    epochs = param['epochs']
    learn_rate = param['learn_rate']
    hidden_nodes = param['hidden_nodes'] ## or whatever "third of the input size" the authors use
    feature_size = param['feature_size'] ## or 13 or 25 or whatever we want. But, it is something we have to pick by hand
                                            ## to recreate paper, we will probably need to try all 3 author values?
        
    ## Separate our testing and training data using a cutoff index
    ## index for the cutoff of our training vs test data
    ## e.g. training data (file names) = processed_imgs[0:train_end_index], test data = processed_imgs[train_end_index:]
    ## this index can be made a "   "   "   "  "   "   "   " " " " " " " " " " " " " " 
    train_dir = f'{processed_images_dir}/train'
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    test_dir = f'{processed_images_dir}/test'
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    train_end_index = param['train_end_index'] ## moved this to param.json
    # train_names = processed_imgs_list[0:train_end_index]
    # test_names = processed_imgs_list[train_end_index:]
    for i, fname in enumerate(processed_imgs_list):
        if i <= train_end_index:
            shutil.move(f'{processed_images_dir}/{fname}', f'{train_dir}/{fname}')
        else:
            shutil.move(f'{processed_images_dir}/{fname}', f'{test_dir}/{fname}')
            
        
    ## read in our labels
    ids_and_labels = pd.read_csv(ids_and_labels_path) 
    
    # Initialize the verbosity printer
    vprinter = VerbosityPrinter(system_verbosity)

    # ## get a (global) mean image vector, and big cov. matrix, PCA matrix
    # ## this will probably take a minute to run
    # mean_vector = dc.mean_image_vec(processed_images_dir)                          ## mean flattened vector of whole dataset
    # vprinter.vprint("mean vector acquired", 2)

    # thetas_mat = dc.matrix_of_thetas(mean_vector, processed_images_dir)            ## matrix of thetas for whole dataset
    # vprinter.vprint("thetas matrix acquired", 2)

    # PCA_matrix = dc.mat_of_thetas_to_pcs(thetas_mat, feature_size)      ## matrix of big C's principle components
    # vprinter.vprint("PCA matrix acquired", 2)
    
    # Compress and label the processed dataset
    vprinter.vprint("Processing training data:", 1)
    train_dataset = GalaxiesDataset(processed_images_dir = train_dir, 
                                    ids_and_labels_path = ids_and_labels_path, 
                                    feature_size = feature_size, 
                                    vprinter = vprinter)
    vprinter.vprint("Processing testing data:", 1)
    test_dataset = GalaxiesDataset(processed_images_dir = test_dir, 
                                    ids_and_labels_path = ids_and_labels_path, 
                                    feature_size = feature_size, 
                                    vprinter = vprinter,                            
                                    train_dataset = train_dataset)

    # ## we can e.g. construct a numpy array with all of our dataset's feature vectors
    #     ## each row = feature vector
    # ## note: if we vary the size of feature vectors, might need to adjust this somehow
    # ## or, just run main.py every time we adjust the feature vector size (since it is a hyper parameter)

    # feature_array = np.zeros((len(ids_and_labels), feature_size), float) ## or int? see what pytorch inevitably complains about 

    # for k in range(len(processed_imgs_list)): 
    #     processed_fname = f'{processed_images_dir}/{ids_and_labels.ID[k]}.jpg'
    #     current_flat_img = dc.flattener(processed_fname)
    #     current_feature_vec = dc.feature_extract(PCA_matrix, current_flat_img, mean_vector)
        
    #     feature_array[k,:] += current_feature_vec
        
    # ## seems a bit weird, last couple look like they are all zeroes? 
    # ## NOTE: (Skye) either the last few images are indistinguishable from the mean image, or I implemented PCA wrong. I'll look into this soon.   
    # ## NOTE: Probably the latter, just checked the last two images, and they look different from each other.
    # ## This^ problem has been resolved. I was only adding feature vectors up to the end of the training data, excluded the test data LOL
    # vprinter.vprint(feature_array[-1,:], 2)
    
    # ## split the feature vectors of each galaxy into training and testing sets
    # train_features_np = feature_array[0:train_end_index,:]
    # test_features_np = feature_array[train_end_index:,:]

    ## WE NEED: some set of INDECIES corresponding to each class, to give as a "label" for our loss function
    ## looks like 'ids_and_labels' just has S and E classes at the moment
    ## this is an ad hoc bit of code for now, should be more generalized in principle
    ## e.g. if the "number of classes" hyperparameter is made > 2
    ## I invite anyone who has a better idea on how to do this to tweak it 
    ## but if we are just doing S and E, need something like:

    # class_labels = np.zeros((len(ids_and_labels)), dtype=np.int64) ## according to my (Ben) A2, pytorch is expecting int64 for loss function
    # for k in range(np.size(class_labels)):
        
    #     ## let spiral galaxies have a label = 0
    #     ## let elliptical galaxies have label = 1
        
    #     ## is this how indexing in Pandas works?
    #     ## i.e. will it recognize that underscore even though the original csv column is "Simple Classification" with a space?
    #     if ids_and_labels['Simple Classification'][k] == 'E':
    #         class_labels[k] = 1
    #     elif ids_and_labels['Simple Classification'][k] == 'S':
    #         continue
    #     else:
    #         raise ValueError("your ad hoc label thing needs more classes")

            
    ## NN stuff - cf. Workshop 2 / assignment 2
    model = Net(feature_size, hidden_nodes, num_class)
    optimizer = optim.SGD(model.parameters(), lr=learn_rate) 
    loss = nn.CrossEntropyLoss()                        ## or, if we are only doing 2 classes, could use BCEloss?
    
    ## this is how i (Ben) structure my training loops in the assignments, not written in stone or anything though, feel free to tweak
    ## I usually "count" how many epochs I have gone through, and use a "while" loop

    # epoch_count = 0
    ind = 0           ## this index will be the first index of the current patch of data
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
        # ## and this is how I usually handle batches
        # ## "ind" is the index of the first row in the batch
        # ## "ind+batch" is the (EXCLUDED) index at the end of the batch
        # ## if this last index does not go past the size of your training data:
        # if ind+batch <= train_end_index:
        #     batch_end = ind+batch   ## then cut the batch off at this (excluded) index
        #     epoch_complete = False  ## and you have not gone through the current epoch completely yet
            
        # ## or if your current batch is near the end of the training data,
        # ## and having a batch size = "batch" would exceed the size of the training data:
        # else:
        #     batch_end = train_end_index  ## then cut the batch off at the end of the training data 
        #     epoch_complete = True        ## after this last batch, you will be finished the current epoch

        # ## note: as of right now, this batch is still a numpy array
        # current_train_batch = train_features_np[ind:batch_end,:]
        
        # ## obscure case but worth throwing in because this has given be problems before
        # ## if your training data size is perfectly divisible by batch size, it can give you 
        # ## a weird batch with 0 size at the end of the training data. This just skips over that case,
        # ## and moves on to the next epoch
        
        # if np.size(current_train_batch) == 0:
        #     ind = 0
        #     epoch_count += 1
        #     continue 
            
        # ## then something like:
        # ## (note: the size of this tensor is [batch size] x [number of classes],
        # ## with its ROWS being the (non-softmaxed) model output PER galaxy

        # NN_output = model.forward(torch.from_numpy(current_train_batch.astype(np.float32)))  ## needs to be float32, forward doesn't like float64
        # ## get the correct labels
        # label_batch = torch.from_numpy(class_labels[ind:batch_end])
        
        # ## compute the loss
        # loss_value = loss(NN_output, label_batch) 
        
        # ## add to the list to plot it
        #     ## or maybe "if [training step number] % 10 = 0 " or something if we don't want too too many
        # ## if we have a lot of epochs, this could be a tracer value for our loss (i.e. just the first batch)
        # if ind == 0:
        #     loss_list.append(loss_value.detach().numpy())
        
        # ## update weights etc
        # optimizer.zero_grad()
        # loss_value.backward() 
        # optimizer.step() 
        
        # #### MORE STUFF HERE #####
        # ## getting something for "accuracy" (how many model predictions have loss value = 0 or =/= 0)
    
        # ## then, if you are at the end of the epoch:
        # if epoch_complete:
        #     ind = 0          ## reset this to start the next batch at the beginning of the training data
        #     epoch_count += 1
        # else:
        #     ind += batch     ## or, if you are not finished the epoch, begin the next batch after your current one
            
    
    ## training loss plot
    # vprinter.vprint(type(loss_list[0]), 2)
    # vprinter.vprint(type(range(epochs)), 2)
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
    plt.savefig('loss_plots.pdf')
    plt.show()
    
    ## Finally, test data
    ## can do it all as one big batch 
    test_loss = model.test(galaxies_data = test_dataset,
                            loss_fn = loss,
                            vprinter = vprinter,
                            batch_size = batch,
                            show_accuracy = True
                            )
    # test_output = model.forward(torch.from_numpy(test_features_np.astype(np.float32)))
    # test_labels = torch.from_numpy(class_labels[train_end_index:])
    # test_loss = loss(test_output, test_labels)
    
    vprinter.vprint(f'TEST DATA LOSS: {test_loss}')
    
    
if __name__ == '__main__':
    main()