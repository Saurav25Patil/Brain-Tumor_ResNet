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

from utils import load_dicom_image,load_dicom_images_3d

def Load_Data(data_directory):
    train_df= pd.read_csv(data_directory+"train_labels.csv")
    train_df['BraTS21ID5']=[format(x,'05d') for x in train_df.BraTS21ID]
    train_df.head(3)

    test = pd.read_csv(
        data_directory+'sample_submission.csv')

    test['BraTS21ID5'] = [format(x, '05d') for x in test.BraTS21ID]
    test.head(3)

    sample_img = dicom.read_file(
    data_directory+"train/00011/FLAIR/Image-400.dcm").pixel_array

    preproc_img = load_dicom_image(data_directory+"train/0011/FLAIR/Image-400.dcm")

    """Sample loading of array of image"""

    fig = plt.figure(figsize=(12,8))
    ax1= plt.subplot(1,2,1)
    ax1.imshow(sample_img,cmap='gray')
    ax1.set_title(f"Original image shape = {sample_img.shape}")
    ax2 = plt.subplot(1,2,2)
    ax2.imshow(preproc_img[:,:,0], cmap="gray")
    ax2.set_title(f"Preproc image shape = {preproc_img.shape}")
    plt.show()

    """Sample loading of an 3D image"""

    sample_seq = load_dicom_images_3d("00011")
    print("Shape of the sequence is:", sample_seq.shape)
    print("Dimension of the 15th image in sequence is:", sample_seq[15].shape)
    fig = plt.figure(figsize=(5,5))
    plt.imshow(np.squeeze(sample_seq[15][:,:,0]), cmap="gray")
    plt.show()

    return train_df,test