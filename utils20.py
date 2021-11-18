import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, LSTM
from keras.datasets import mnist
from keras import regularizers, initializers, optimizers
import os
import datetime
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import imagesize
import tensorflow as tf
import seaborn as sns
    
def matrix_confusion(model, x, y):
    """Returns a confusion matrix in Seaborn heatmap style.
    
    model: model 
    x: X_val (X validation) 
    y: y_val (y validation)
    
    """
    y_predict_test = model.predict(x)
    y_true_test = y
    res_test = tf.math.confusion_matrix(y_true_test, y_predict_test)
    res_test
    
    return sns.heatmap(res_test, annot=True, fmt='g')
    
