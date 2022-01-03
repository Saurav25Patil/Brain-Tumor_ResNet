import glob
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

from config import IMAGE_SIZE, MRI_TYPE, NUM_IMAGES, SCALE

data_directory = '../input/rsna-miccai-brain-tumor-radiogenomic-classification/'

def load_dicom_image(path,img_size=IMAGE_SIZE,scale=SCALE):
    """loads dicom images and applies preprocessing steps like crop, resize and denoising filter

    Args:
        path (String): path to the DCIM image file to load
        img_size (int, optional): [image size desired for resizing]. Defaults to IMAGE_SIZE.
        scale (Float, optional): [Desired scale for the cropped image]. Defaults to SCALE.

    Returns:
        array: returns array of images
    """

    #Load a single image
    img = dicom.read_file(path).pixel_array
    #Crop the image
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]

    #Resize the image
    img = cv2.resize(img,(img_size,img_size))

    #Convert into a 3D array
    img = np.repeat(img[...,np.newaxis],3,-1)

    return img

def load_dicom_images_3d(scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type=MRI_TYPE,
    split= "train"):

    """loads an ordered sequence of x preprocessed images starting from the central image of each folder

    Args:
        - scan_id : String
        ID of the patient to load.
    - num_imgs : Integer
        Number of desired images of the 
        sequence.
    - img_size : Integer
        Image size desired for resizing.
    - scale : Float
        Desired scale for the cropped image
    - mri_type : String
        Type of scan to load (FLAIR, T1w, 
        T1wCE, T2).
    - split : String
        Type of split desired : Train or Test
    Returns:
        3d array: returns 3d array of images
    """

    
    
    files = sorted(glob.glob(f"{data_directory}{split}/{scan_id}/{mri_type}/*.dcm"),
        key= lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+',var)])

    middle = len(files)//2
    num_imgs2 = num_imgs//2
    p1=max(0,middle-num_imgs2)
    p2 = min (len(files),middle+num_imgs2)
    img3d = np.stack([load_dicom_image(f) for f in files[p1:p2]])

    if img3d.shape[0]<num_imgs:
        n_zero = np.zeros((num_imgs-img3d.shape[0],img_size,img_size,3))
        img3d=np.concatenate((img3d,n_zero),axis=0)
    
    return img3d
