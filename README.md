# PHYS_449_project

Repository for our PHYS 449 group project

## Dataset format

Our dataset consists of a folder with all of the raw images (raw_images) and two CSV files:

- ids_and_labels.csv contains the object IDs, the binary classifications, and the full classifications 
- specific_ids_and_labels.csv contains the object IDs, the multiclass classifications, and the full classifications

## Project Dependencies

### Python

We are using Python 3.10.5

### Create a new venv (virtual environment)

```sh
python -m venv group_proj_env
```

Note: you could also use conda, or another package manager to create and manage virtual environments.

### Activate a venv

1. Using VSCode's file explorer, go to `group_proj_env/Scripts/`
2. Right click on the file named `activate`(should be the first one)
3. Click on "Copy Relative Path"
4. Paste this path in the command line and hit enter

### Packages Required

- numpy
- opencv
- matplotlib
- pandas
- requests
- scikit-learn
- pytorch
- nptyping

### Use this command to install the required packages

```sh
pip install -r requirements.txt
```

### sdss package credits
The "sdss" folder in our "src" folder comes from the "sdss" package created by Behrouz Safari (https://github.com/behrouzz/sdss). Used under an MIT open source license 

## Running `main.py`

To run main.py, simply paste this in the command line and hit enter:
```
python main.py
```

## json, hyperparameters

The hyperparameters can be found in the "param/param.json" file. It is split up into 3 sections/subdictionaries

#### optim

- epochs: number of times the traning dataset is iterated over
- learn_rate: the learning rate used by the NN's optimizer
- binary_threshold: the "cutoff" value used when processing our images into binary images. Pixel values below this value = 0, pixel values above it = 1

#### model

- hidden_nodes: the size of the hidden layer inside our NN
- feature_size: the dimension of the feature vector inputted into the NN, i.e. the number of principle components kept from the covariance matrix for the whole processed dataset
- batch: number of feature vectors trained on before updating model weights during training
- train_end_index: the size of our training dataset

#### class_label_mapping

Assigns integer values for galaxy classes/subclasses being considered 