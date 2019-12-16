import numpy as np 
import keras
from keras.layers import Conv2D,MaxPooling2D,Dropout,LeakyReLU,concatenate,ZeroPadding2D,BatchNormalization,Conv2DTranspose
from keras.layers import Add
from keras.models import Model
from keras.layers import Input, Dense
import os 
import cv2 
from matplotlib.pyplot import imshow,title ,show
from keras.applications import Xception
from keras import backend as K
from keras.losses import BinaryCrossentropy  as binary_crossentropy
from keras.callbacks import ModelCheckpoint,Callback
from keras.optimizers import Adam
from matplotlib import pyplot as plt




for idx in range(100):

    img_test = X_val[idx]
    img_test_ary = np.reshape(img_test,(1,512,512,3))
    mask_test  = model.predict(img_test_ary)
    mask_test = np.reshape(mask_test,(512,512))
    mask_test = np.around(mask_test)
    #mask_test = mask_test > 0.95
    mask_test = mask_test.astype('uint8')
    #imshow(mask_test)



    img_test = X_val[idx]
    #img_test_ary = np.reshape(img_test,(1,512,512,3))
    #mask_true  = model.predict(y_val[idx])
    mask_true = np.reshape(y_val[idx],(512,512))
    mask_true = np.around(mask_true)
    mask_true = mask_true.astype('uint8')
    #imshow(mask_true)


#     f, axarr = plt.subplots(1,3,figsize=(10,10))
#     axarr[0].imshow(X_val[idx])
#     axarr[1].imshow(mask_true)
#     axarr[2].imshow(mask_test)

    #Superimposing the images
    #gray_img = cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)
    #rgb_mask = cv2.cvtColor(mask_pred,cv2.COLOR_GRAY2RGB)
    #resized_img = cv2.resize(img,(512,512))
    #gray_img = cv2.resize(gray_img,(512,512))
    #super_impose_pred_img = cv2.addWeighted(gray_img,0.1,mask_test,8,0.2)
    
#     x, y, w, h = cv2.boundingRect(mask_test)
#     rect1 = cv2.rectangle(img_test.copy(),(x,y),(x+w,y+h),(255,0,0),3)
    
    #Superimposing the images
    gray_img = cv2.cvtColor(img_test,cv2.COLOR_BGR2GRAY)
    #rgb_mask = cv2.cvtColor(mask_pred,cv2.COLOR_GRAY2RGB)
    #resized_img = cv2.resize(img,(512,512))
    gray_img = cv2.resize(gray_img,(512,512))
    super_impose_true_img = cv2.addWeighted(gray_img,0.1,mask_true,8,0.2)
    
    rect2 = img_test.copy()
    contours,_ = cv2.findContours(mask_test.copy(), 1, 1) # not copying here will throw an error
    for i,cntr in enumerate(range(len(contours))):
        rect = cv2.minAreaRect(contours[i]) # basically you can feed this rect into your classifier
        (x,y),(w,h), a = rect # a - angle
    
        box = cv2.boxPoints(rect)
        box = np.int0(box) #turn into ints
        rect2 = cv2.drawContours(rect2,[box],0,(255,0,0),3)

    
       #Plotting the mask and image
    f, axarr = plt.subplots(1,3,figsize=(30,30))
    axarr[0].set_title('True Image {0}'.format(idx),fontsize=30)
    axarr[0].imshow(img_test,cmap='gray')
    axarr[1].set_title('True Box {0}'.format(idx),fontsize=30)
    axarr[1].imshow(super_impose_true_img,cmap='gray')
    #axarr[1].imshow(mask_pred,cmap='gray')
    axarr[2].set_title('Predicted Box {0}'.format(idx),fontsize=30)
    axarr[2].imshow(rect2)
    
    show()
