from pathlib import Path
import imagesize
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import tensorflow as tf

def pic_count(path, folder_name):
    ''' Returns a visualization of the picture counts for image folders by type.
    
    path: The path to the folder containing the images. 
    folder_name: name of the folder as a string. This will set the title of the visualization. 
    
    '''
    dict = {'Normal' :len(os.listdir(path + '/NORMAL')), 'Pneumonia':len(os.listdir(path + '/PNEUMONIA'))}
    fig = px.bar(dict.keys(), 
                 dict.values(), 
                 title=folder_name, 
                 color=list(dict.keys()), 
                 orientation='h', 
                 color_discrete_sequence=px.colors.sequential.ice_r, 
                 template='plotly_dark', 
                 width=800,
                 height=500)
    fig.show()
    
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
    
def images_to_df(root1_, root2_, image_type1, image_type2):
    """ Returns a concatenated DataFrame containing columns for FileName, Size, Width, Aspect Ratio, and Image Type.
    Some of this code was taken from https://medium.com/analytics-vidhya/how-to-pick-the-optimal-image-size-for-training-convolution-neural-network-65702b880f05 that grabs the Width and Height of the images. 
    
    root1: first path to image folder.
    root2: second path to image folder. 
    image_type1: Image Type from the first folder.
    image_type2: Image Type from the second folder.
    
    """ 
    imgs1 = [img.name for img in Path(root1_).iterdir() if img.suffix == ".jpeg"]
    img_meta1 = {}
    for f in imgs1: img_meta1[str(f)] = imagesize.get(root1_+f)

    # Convert it to Dataframe and compute aspect ratio
    data1 = pd.DataFrame.from_dict([img_meta1]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
    data1[["Width", "Height"]] = pd.DataFrame(data1["Size"].tolist(), index=data1.index)
    data1["Aspect Ratio"] = round(data1["Width"] / data1["Height"], 2)
    data1["Type"] = image_type1
    
    #same for second root
    imgs2 = [img.name for img in Path(root2_).iterdir() if img.suffix == ".jpeg"]
    img_meta2 = {}
    for f in imgs2: img_meta2[str(f)] = imagesize.get(root2_+f)

    # Convert it to Dataframe and compute aspect ratio
    data2 = pd.DataFrame.from_dict([img_meta2]).T.reset_index().set_axis(['FileName', 'Size'], axis='columns', inplace=False)
    data2[["Width", "Height"]] = pd.DataFrame(data2["Size"].tolist(), index=data2.index)
    data2["Aspect Ratio"] = round(data2["Width"] / data2["Height"], 2)
    data2["Type"] = image_type2

    #Concat DataFrames
    data = pd.concat([data1, data2], axis=0)
    
    return data
    
def visualize_training_results(history):
    '''
    From https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    
    Input: keras history object (output from trained model)
    '''
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    fig.suptitle('Model Results')

    # summarize history for accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_ylabel('Accuracy')
    ax1.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_ylabel('Loss')
    ax2.legend(['train', 'test'], loc='upper left')
    
    plt.xlabel('Epoch')
    plt.show()
    
def plot_performance(hist):
    """ Returns 4 plots comparing Training and Validation data. 
    First plot returns training and validation accuracy. 
    Second plot returns training and validation loss. 
    Third plot returns training and validation F1-Scores. 
    Fourth plot returns training and validation recall scores. 
    
    hist: input history model containing train images, labels, and validation data. t"""
    
    hist_ = hist.history
    epochs = hist.epoch
    
    plt.plot(epochs, hist_['accuracy'], label='Training Accuracy')
    plt.plot(epochs, hist_['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    
    plt.plot(epochs, hist_['loss'], label='Training loss')
    plt.plot(epochs, hist_['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    recall = np.array(hist_['recall'])
    precision = np.array(hist_['precision'])
    val_recall = np.array(hist_['val_recall'])
    val_precision = np.array(hist_['val_precision'])
    plt.figure()
    
    plt.plot(epochs, 
             2*((recall * precision)/(recall + precision)), 
             label='Training f1')
    plt.plot(epochs, 
             2*((val_recall * val_precision)/(val_recall + val_precision)), 
             label='Validation f1')
    plt.title('Training and validation F1-Score')
    plt.legend()
    plt.figure()
    
    plt.plot(epochs, recall, label = "Training Recall")
    plt.plot(epochs, val_recall, label = "Validation Recall")
    plt.title("Training and Validation Recall Scores")
    plt.legend()
    plt.figure()
    
    plt.show()
    
def confusion_matrix(estimator, x, y):
    """Returns a confusion matrix in Seaborn heatmap style.
    
    estimator: model 
    x: X_val (X validation) 
    y: y_val (y validation)
    
    """
    y_predict_test = estimator.predict(x)
    y_true_test = y
    res_test = tf.math.confusion_matrix(y_true_test, y_predict_test)
    res_test
    
    return sns.heatmap(res_test, annot=True, fmt='g')


    