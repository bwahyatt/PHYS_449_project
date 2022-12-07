import sdss # https://github.com/behrouzz/sdss
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from verbosity_printer import VerbosityPrinter

def dataset_generate(galaxy_zoo_csv: str, N: int, imgs_output_dir: str, labels_csv: str, vprinter: VerbosityPrinter = None) -> None:
    '''
    Generates our dataset of galaxies by extracting the relevant information from Galaxy Zoo

    Args:
        galaxy_zoo_csv (str): Filepath to the Galaxy Zoo .csv file
        N (int): The number of galaxies to generate
        imgs_output_dir (str): Filepath to output all the galaxy images
        labels_csv (str): Filepath to output the labels and IDs .csv file to
    '''
    
    if vprinter is None:
        vprinter = VerbosityPrinter(1)
    
    try:
        pdir = '..'
        galaxy_zoo = pd.read_csv(f'{pdir}/{galaxy_zoo_csv}') #import galaxy zoo dataset
    except:
        pdir = '.'
        galaxy_zoo = pd.read_csv(f'{pdir}/{galaxy_zoo_csv}') #import galaxy zoo dataset
    
    mini_zoo = galaxy_zoo.iloc[:N] #small dataset of the number of galaxies we want to generate

    #make lists to store the IDs and classifications
    id_list = []
    class_list = []
    class_simple_list = []


    for x in tqdm(range(len(mini_zoo)), desc = 'Retrieving and generating data from SDSS', disable = vprinter.system_verbosity == 0):
        ra = mini_zoo.iloc[x]['ra'] #Right ascension
        dec = mini_zoo.iloc[x]['dec'] #Declination 
        class_full = mini_zoo.iloc[x]['gz2_class'] #full classification
        class_simple = mini_zoo.iloc[x]['gz2_class'][0] #Simple classification, i.e. spiral or elliptical. From first letter of classification
        reg = sdss.Region(ra, dec, fov=0.033)
        
        obj_id = reg.nearest_objects().iloc[0]['objID'] #ID of the object nearest to the region, AKA the object we are looking at
        ph = sdss.PhotoObj(obj_id) ## this argument is the SDSS ObjID
        ph.download()

        id_list.append(obj_id)

        class_list.append(class_full)
        class_simple_list.append(class_simple)


        img_data = ph.cutout_image(scale=0.1) #returns np array of image

        cv2.imwrite(f'{pdir}/{imgs_output_dir}/{obj_id}.jpg', img_data) #write the array to a jpg file


    #Put IDs and classifications into a dataframe
    labels = pd.DataFrame(data={'ID': id_list, 'Simple Classification': class_simple_list, 'Full Classification': class_list})
    labels.to_csv(f'{pdir}/{labels_csv}.csv')

if __name__ == '__main__':
    dataset_generate(galaxy_zoo_csv = 'galaxy_zoo_shortened.csv',
                     N = 450,
                     imgs_output_dir = 'raw_images',
                     labels_csv = 'ids_and_labels')