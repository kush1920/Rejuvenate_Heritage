#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#sess = tf.compat.v1.Session(config=config)
sess = tf.compat.v1.InteractiveSession(config=config)
import keras.backend.tensorflow_backend as tfback
def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings)

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus
tfback._get_available_gpus()
tf.config.list_logical_devices()
from copy import deepcopy
import cv2
import numpy as np
from PCONV_UNET import *

# In[ ]:


def predict_final(img , mask):
    model = PConvUnet(256 , 512 , vgg_weights=None)
    model.load('gan9.h5', train_bn=False)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img , (512 , 256))
    
    mask = cv2.bitwise_not(mask)
    mask = cv2.resize(mask , (512 , 256))
    masked_img = deepcopy(img)
    masked_img[mask==0] = 255
    
    x = [np.expand_dims(masked_img , 0)/255.0 , np.expand_dims(mask , 0)/255.0]
    
    restored = model.predict(x)[0]
    restored = np.array(restored[:,:,:]*255 , np.uint8)
    
    return restored
