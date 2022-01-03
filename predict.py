import glob
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

from utils import load_dicom_images_3d
from config import NUM_IMAGES,MRI_TYPE

def Prediction(test,base_resnet,best_kfold_model):
    #Prediction

    X_test = test['BraTS21ID5'].values
    test_listMatrix = []
    for i, patient in enumerate(tqdm(X_test)):
        test_listVectors = []
        test_sequence = load_dicom_images_3d(scan_id=str(patient),mri_type=MRI_TYPE,split="test")
        for j in range(len(test_sequence)):
            img = test_sequence[j]
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.resnet50.preprocess_input(img)
            img_vector = base_resnet.predict(img)
            test_listVectors.append(np.array(img_vector))
        
        test_PatientMatrix = np.stack(test_listVectors)
        test_listMatrix.append(test_PatientMatrix)

    print(f"Number of test patient matrix: {len(test_listMatrix)}")
    print(f"Test patient matrix shape: {test_listMatrix[0].shape}")

    test_dataset = tf.data.Dataset.from_tensor_slices(test_listMatrix)
    len(test_dataset)

    final_model = keras.models.load_model(best_kfold_model)
    predict = final_model.predict(test_dataset)
    print(predict.shape)

    predict = predict[:,0,0]
    final_predict = []
    for i in range(len(test_listMatrix)):
        i+=1
        final_predict.append(round(predict[((i-1)*NUM_IMAGES):(NUM_IMAGES*i)].mean(),3))
    submission = test[["BraTS21ID","MGMT_value"]]
    submission["MGMT_value"] = final_predict
    submission.to_csv('submission.csv',index=False)
    submission.head(5)

    plt.figure(figsize=(8, 8))
    plt.hist(submission["MGMT_value"])
    plt.title("Predicted probabilites distribution on test set", 
            fontsize=18, color="#0b0a2d")
    plt.show()