## how to use Behrouz Safari's SDSS package for our purposes
## references: 
## https://astrodatascience.net/sdss-objects/
## https://github.com/behrouzz/sdss
    ## I have copied the guts of this repo into ours so imports etc should all work

import sdss
import numpy as np
import cv2

## create an instance of a "PhotoObj" class
## note: these objects have a ton of custom attributes, methods, arguments etc. to play around with

ph = sdss.PhotoObj(1237648720693755918) ## this argument is the SDSS ObjID (this example one is lifted from the blogpost above)
ph.download()                           ## this will actually download the image from SDSS data
ph.show()                               ## this is exactly what you think is (showing the image)

img_data = ph.cutout_image(scale=0.1)   ## this returns an np array of the image (I think with RGB channels)
                                        ## not 100% sure which "scales" etc are best to use,
                                        ## "cutout_image" has many optional arguments
        
cv2.imwrite('SDSS_image_test.jpg', img_data) ## writes the np array to a jpg with opencv
                                             ## I (Ben) am not sure yet if SDSS package has direct way of writing out images 
                                             ## this will do the trick in any case   
print('done') ## sanity check

#### SO TO GENERATE A DATASET: WE SHOULD MAKE A SCRIPT WITH SOMETHING LIKE:

## read in a csv with SDSS ObjIDs and classification labels (e.g. from Galaxy Zoo [??] )
## iterate over the ObjIDs
## create a PhotoObj object of each objID (sdss package)
## download image, save it as a jpg to some folder of raw images

## then: run a script that processes the raw images using our '04 paper method (see "image_processor_script.py")
