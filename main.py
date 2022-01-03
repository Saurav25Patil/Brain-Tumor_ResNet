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

from load_data import Load_Data
from model import Load_pretrained_model
from predict import Prediction
from train import Train


def main():
    data_directory = '../input/rsna-miccai-brain-tumor-radiogenomic-classification/'
    train_df, test=Load_Data(data_directory)

    listMatrix,base_resnet,train,X_train,y_train=Load_pretrained_model(train_df)

    best_kfold_model=Train(listMatrix,y_train=y_train)
    Prediction(test,base_resnet,best_kfold_model)



if __name__=="__main__":
    main()

