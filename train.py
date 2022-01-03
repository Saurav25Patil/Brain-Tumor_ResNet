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

from utils import NUM_IMAGES


def Train(listMatrix,y_train):
    # Apply LSTM for classification

    model_input_dim = listMatrix[0].shape[2]

    def get_sequence_model():
        """Defines the LSTM model architecture
        - LSTM
        - Dropout with a probability of 0.2
        - ReLu activation layer with 100 inputs
        - Sigmoid activation layer with 1 input
        """
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(100,input_shape=(NUM_IMAGES,model_input_dim),return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Dense(100,activation='relu'))
        model.add(keras.layers.Dense(1,activation='sigmoid'))
        return model

    #Create k-fold cross validator and training

    inputs = np.array(listMatrix)
    targets = np.array(y_train).astype('float32').reshape((-1,1))

    num_folds=5

    #Defining the k-fold classifier
    kfold = KFold(n_splits=num_folds,shuffle=True)

    history={}
    fold_no=1

    for train_df, valid_df in kfold.split(inputs,targets):

        train_dataset = tf.data.Dataset.from_tensor_slices((inputs[train_df] , targets[train_df]))
        valid_dataset = tf.data.Dataset.from_tensor_slices((inputs[valid_df] , targets[valid_df]))

        model = get_sequence_model()
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics='accuracy')

        # Define callbacks.
        model_save = ModelCheckpoint(f'Brain_lstm_kfold_{fold_no}.h5',
                                    save_best_only = True,
                                    monitor = 'val_accuracy',
                                    mode = 'max', verbose = 1)
        early_stop = EarlyStopping(monitor = 'val_accuracy',
                                patience = 25, mode = 'max', verbose = 1,
                                restore_best_weights = True)

        print('--------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        epochs = 200
        history[fold_no] = model.fit(
            train_dataset,
            validation_data=valid_dataset, 
            epochs=epochs, 
            batch_size=32,
            callbacks = [model_save, early_stop])
        
        # Increase fold number
        fold_no += 1

    #Results of the training

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    ax = ax.ravel()

    for fold in history:
        for i, metric in enumerate(["accuracy","loss"]):
            ax[i].plot(history[fold].history[metric], label="train "+str(fold))
            ax[i].plot(history[fold].history["val_" + metric], linestyle="dotted", label="val "+str(fold))
            ax[i].set_title("Model {}".format(metric))
            ax[i].set_xlabel("epochs")
            ax[i].set_ylabel(metric)
            ax[i].legend()


    kfold_results = pd.DataFrame(columns=["Fold","Mean_Loss","Mean_Accuracy"])
    key=[]
    mean_acc=[]
    mean_loss=[]
    for fold in history:
        key.append(fold)
        mean_loss.append(np.mean(history[fold].history["val_loss"]))
        mean_acc.append(np.mean(history[fold].history["val_accuracy"]))

    kfold_results["Fold"] = key
    kfold_results["Mean_Loss"] = mean_loss
    kfold_results["Mean_Accuracy"] = mean_acc
    kfold_results["Rank_Ratio"] = (kfold_results["Mean_Loss"] - kfold_results["Mean_Accuracy"])
    kfold_results = kfold_results.sort_values("Rank_Ratio", ascending=True)
    kfold_results  

    #Best model

    best_kfold_model = './Brain_lstm_kfold_' + str(kfold_results.Fold.values[0]) + '.h5'
    print(f"The best model selected is {best_kfold_model}")

    return best_kfold_model
