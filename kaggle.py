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

# Load the Data

data_directory = '../input/rsna-miccai-brain-tumor-radiogenomic-classification/'

train_df= pd.read_csv(data_directory+"train_labels.csv")
train_df['BraTS21ID5']=[format(x,'05d') for x in train_df.BraTS21ID]
train_df.head(3)

test = pd.read_csv(
    data_directory+'sample_submission.csv')

test['BraTS21ID5'] = [format(x, '05d') for x in test.BraTS21ID]
test.head(3)

IMAGE_SIZE = 240
SCALE = .8
NUM_IMAGES = 64
MRI_TYPE = "FLAIR"


#Load a single image

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


#sample for a random patient

sample_img = dicom.read_file(
    data_directory+"train/00011/FLAIR/Image-400.dcm").pixel_array

preproc_img = load_dicom_image(data_directory+"train/0011/FLAIR/Image-400.dcm")

fig = plt.figure(figsize=(12,8))
ax1= plt.subplot(1,2,1)
ax1.imshow(sample_img,cmap='gray')
ax1.set_title(f"Original image shape = {sample_img.shape}")
ax2 = plt.subplot(1,2,2)
ax2.imshow(preproc_img[:,:,0], cmap="gray")
ax2.set_title(f"Preproc image shape = {preproc_img.shape}")
plt.show()


# load a sequence of 64

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

#Sample loading of an 3D array of images

sample_seq = load_dicom_images_3d("00011")
print("Shape of the sequence is:", sample_seq.shape)
print("Dimension of the 15th image in sequence is:", sample_seq[15].shape)
fig = plt.figure(figsize=(5,5))
plt.imshow(np.squeeze(sample_seq[15][:,:,0]), cmap="gray")
plt.show()

# Load pre trained ResNet34 model

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