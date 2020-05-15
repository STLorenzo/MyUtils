# HTTP and URL Libraries
import urllib3 # URL request library
import certifi # Certifications library for secure url requests

# Path Libraries
from pathlib import Path # Path manipulation
import shutil # high-level operations on files and collections of files


# System Libraries
import os # OS library
import zipfile # zip manipulation library
from datetime import datetime # Datetime data manipulation Library
from collections import Counter # Dict manipulations
import re # Regular expressions library
import glob # Unix style pathname pattern expansion
from tqdm import tqdm # Progress bar library
import random # random number generator library


# Data science Libraries
import pandas as pd # Data import, manipulation and processing 
import numpy as np # Vector - Matrix Library
import pickle # Object serializing module.


# Data visualization Libraries
import matplotlib.pyplot as plt # Graph making Library
import seaborn as sns # graphics library 

%matplotlib inline # Shows matplotlib graphs in the notebook


# --------------------------------------Machine Learning Libraries------------------------
# TensorFlow
import tensorflow as tf # Tensorflow Library

# Keras Libraries
from keras.models import Sequential # Feedfordward Neural Network model

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # keras image manipulation

from keras.layers import Conv2D # Convolutional layer uses a grid to learn and extract features of the image
from keras.layers import MaxPooling2D # MaxPooling in the Convolutional Window picks the highest value for the feature map

from keras.layers import Dense # Fully connected layer
from keras.layers import Dropout # Dropout layer which doesn't update values in training for the proportion given.
from keras.layers import Flatten # Layer which flattens a matrix MxN into a 1x(M*N) array
from keras.layers import Activation  # Activation Layer

from keras.callbacks.callbacks import Callback # Custom callback creation library
from keras.callbacks.callbacks import ModelCheckpoint
from keras.models import load_model # function that loads a previously stored model


# Sklearn Libraries
from sklearn.svm import SVC # Support vector machine model
from sklearn.multiclass import OneVsRestClassifier # multiclass classifier
from skimage.io import imread # Scikit image library: Reads an image
from skimage.transform import resize # Scikit image library: Resizes an image


# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import roc_curve,roc_auc_score


# Style Libraries
from IPython.display import Markdown, display # Style output display in jupyter notebook

# PyTorch
import torch  # machine learning framework using tensors


