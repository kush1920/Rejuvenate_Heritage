#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
import keras.backend.tensorflow_backend as tfback
def _get_available_gpus():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus
tfback._get_available_gpus()
tf.config.list_logical_devices()
import keras
from keras.models import model_from_json
keras.backend.clear_session()
import numpy as np
import cv2


# In[ ]:


def predict_mask(img):
    with open('mask.json', 'r') as json_file:
        json_savedModel= json_file.read()
    model = model_from_json(json_savedModel)
    model.load_weights('unet_weights.h5')
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    
    img = cv2.resize(img , (768,512))
    
    mask = model.predict(np.expand_dims(img , 0)/255.0)
    mask = (mask > 0.4) *255.0
    mask = np.reshape(mask , (mask.shape[1] , mask.shape[2])).astype('uint8')
    mask = cv2.cvtColor(mask , cv2.COLOR_GRAY2BGR)
    return mask
