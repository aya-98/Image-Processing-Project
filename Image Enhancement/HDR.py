import cv2
import numpy as np
import os



path = 'images'

path2= 'C:/Users/DELL/OneDrive/Desktop/Image-proceesing-project/Image Enhancement/output'

times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
  
filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

images = []

for img in filenames:


    curimg = cv2.imread(f'{path}/{img}')

    images.append(curimg)
  
  
  
  
# Align input images
alignMTB = cv2.createAlignMTB()

alignMTB.process(images, images)



# Obtain Camera Response Function (CRF)
  
calibrateDebevec = cv2.createCalibrateDebevec()
responseDebevec = calibrateDebevec.process(images, times)
  


# Merge images into an HDR linear image
#print("Merging images into one HDR image ... ")
mergeDebevec = cv2.createMergeDebevec()
hdrDebevec = mergeDebevec.process(images, times, responseDebevec)



# # Tonemap using Drago's method to obtain 24-bit color image

tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
ldrDrago = tonemapDrago.process(hdrDebevec)
ldrDrago = 3 * ldrDrago

cv2.imwrite(os.path.join(path2 , "ldr-Drago.jpg"), ldrDrago * 255)
  
  
# # Tonemap using Reinhard's method to obtain 24-bit color image
#print("Tonemaping using Reinhard's method ... ")
tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
ldrReinhard = tonemapReinhard.process(hdrDebevec)

cv2.imwrite(os.path.join(path2 ,"ldr-Reinhard.jpg"), ldrReinhard * 255)

#print("saved ldr-Reinhard.jpg")

  
# # Tonemap using Mantiuk's method to obtain 24-bit color image
tonemapMantiuk = cv2.createTonemapMantiuk(2.2,0.85, 1.2)
ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
ldrMantiuk = 3 * ldrMantiuk

cv2.imwrite(os.path.join(path2 ,"ldr-Mantiuk.jpg"), ldrMantiuk * 255)
