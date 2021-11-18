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


def plotly_eda(data, title1, title2, title3): 
    '''Returns three Plotly visuals including: 
    
    1. Scatter plot of Height and Width (pixels)
    2. Histogram of distrubition Height vs Width, 
    3. Histogram of distribution of Width vs Height.
    
    data: DataFrame with a Height, Width, and Type column. 
    title1: title of first scatter plot. 
    title2: title of first histogram. 
    title3: title of second histogram. 
    
    '''
    fig = px.scatter(data, 
                     x='Width', 
                     y='Height',
                     color='Type',
                     color_discrete_sequence=px.colors.sequential.ice_r,
                     template='plotly_dark',
                     opacity=.4,
                     title=title1
                     )

    fig2 = px.histogram(train_img_sizes_df, 
                        x='Width', 
                        y='Height',  
                        color='Type',
                        color_discrete_sequence=
                        px.colors.sequential.ice_r, 
                        template='plotly_dark',
                        title=title2
                       )

    fig3 = px.histogram(train_img_sizes_df, 
                        x='Height', 
                        y='Width',
                        color='Type',
                        color_discrete_sequence=
                        px.colors.sequential.ice_r, 
                        template='plotly_dark',
                        title=title3
                       )
    fig.show()
    fig2.show()
    fig3.show()
    
def confusion_matrix(model, x, y):
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