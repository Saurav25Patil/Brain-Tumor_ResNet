import math
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import seaborn as sns
import tensorflow as tf
from numpy.core.defchararray import not_equal
from pydicom import sequence
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.python.ops.gen_array_ops import shape
from tqdm.notebook import tqdm

from config import IMAGE_SIZE,MRI_TYPE
from utils import load_dicom_images_3d


def Load_pretrained_model(train_df):
    base_resnet = keras.applications.ResNet50(
        weights=None,
        pooling='avg',
        input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
        include_top=False
    )

    base_resnet.load_weights(
        '../input/resnet-imagenet-weights/base_resnet_imagenet.h5')


    base_resnet.trainable = False

    # Create a matrix of vector base on ResNet50 for each patient sequence.

    train = train_df[['BraTS21ID5','MGMT_value']]
    X_train = train['BraTS21ID5'].values
    y_train = train['MGMT_value'].values


    listMatrix=[]
    for i,patient in enumerate(tqdm(X_train)):
        listVectors=[]
        sequence = load_dicom_images_3d(scan_id=str(patient),mri_type=MRI_TYPE)
        for j in range(len(sequence)):
            img = sequence[j]
            img = np.expand_dims(img,axis=0)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            img_vector=base_resnet.predict(img)
            listVectors.append(np.array(img_vector))

        PatientMatrix = np.stack(listVectors)
        listMatrix.append(PatientMatrix)

    #Check the shape of the matrix

    print(f"Number of patient matrix: {len(listMatrix)}")
    print(f"Patient matrix shape: {listMatrix[0].shape}")

    np.array(listMatrix, dtype=object).shape

    return listMatrix,base_resnet,train,X_train,y_train