import sdss
import numpy as np
import cv2
import pandas as pd

galaxy_zoo = pd.read_csv('../sdss_shortened.csv')
mini_zoo = galaxy_zoo.iloc[:10] #small dataset to test this out on. We can increase the size to whatever we want


for x in range(len(mini_zoo)):
    ra = mini_zoo.iloc[x]['ra'] #Right ascension
    dec = mini_zoo.iloc[x]['dec'] #Declination
    reg = sdss.Region(ra, dec, fov=0.033)

    id = reg.nearest_objects().iloc[0]['objID'] #ID of the object nearest to the region, AKA the object we want
    ph = sdss.PhotoObj(id) ## this argument is the SDSS ObjID
    ph.download()

    img_data = ph.cutout_image(scale=0.1) #returns np array of image
    cv2.imwrite('../raw_images/{}.jpg'.format(id), img_data) #write the array to a jpg file