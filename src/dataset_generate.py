import sdss
import numpy as np
import cv2
import pandas as pd

galaxy_zoo = pd.read_csv('../sdss_shortened.csv') #import galaxy zoo dataset
mini_zoo = galaxy_zoo.iloc[:310] #small dataset to test this out on. We can increase the size to whatever we want

#make lists to store the IDs and classifications
id_list = []
class_list = []
class_simple_list = []


for x in range(len(mini_zoo)):
    ra = mini_zoo.iloc[x]['ra'] #Right ascension
    dec = mini_zoo.iloc[x]['dec'] #Declination 
    class_full = mini_zoo.iloc[x]['gz2_class'] #full classification
    class_simple = mini_zoo.iloc[x]['gz2_class'][0] #Simple classification, i.e. spiral, elliptical, or irregular. From first letter of classification
    reg = sdss.Region(ra, dec, fov=0.033)
    
    ## changing 'id' variable to 'obj_id', apparently 'id' is also a python method
    obj_id = reg.nearest_objects().iloc[0]['objID'] #ID of the object nearest to the region, AKA the object we are looking at
    ph = sdss.PhotoObj(obj_id) ## this argument is the SDSS ObjID
    ph.download()

    id_list.append(obj_id)

    class_list.append(class_full)
    class_simple_list.append(class_simple)


    img_data = ph.cutout_image(scale=0.1) #returns np array of image

    cv2.imwrite('../raw_images/{}.jpg'.format(obj_id), img_data) #write the array to a jpg file


#Put IDs and classifications into a dataframe
labels = pd.DataFrame(data={'ID': id_list, 'Simple Classification': class_simple_list, 'Full Classification': class_list})
labels.to_csv('../ids_and_labels.csv')