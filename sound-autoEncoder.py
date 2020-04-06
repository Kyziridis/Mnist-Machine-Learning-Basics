#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:31:51 2018

@author: dead
"""

import sys
import random
import numpy as np
import keras
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from keras.layers import Input, Dense
from keras.models import Model
from IPython.display import clear_output
from keras import regularizers

#################################################################################
# Provide your own path for the data

path = "/home/dead/Documents/API/Birds/Project"
# Setting working directory
os.chdir(path)
# Import truth csv
my_data = np.matrix(np.genfromtxt('warblrb10k_public_metadata.csv', delimiter=',' , dtype=str , skip_header=1 ))

d = "/home/dead/Documents/API/Birds/train150"        
# Set directory for wav files
os.chdir(d)

def Loading():
    # Import wavs
    r = []
    name = []
    i = 1
    print("\nImporting wav files...")    
    for filename in tqdm(os.listdir(os.getcwd())):    
        #pbar.update()
        x,sr = librosa.load(filename)         
        r.append(x[0:21167])
        #r.append(x)
        name.append(filename.split('.')[0])
        i = i+1
       
    # Make raw as an np.array
    raw = np.asarray(r)        
    # Make truth as groundtruth with labels of wavs
    ind = np.where(my_data[:,0] == name)[0] # find indexes of our wav in my_data
    truth = np.matrix(my_data[ind]) # make truth only with these indexes
    # Fix the order of truth
    tmp = np.argsort(np.where(truth[:,0] == name)[1]) # index the mapping 
    truth = truth[tmp] # set the actural order
    return raw, truth, sr


# Function for plotting waves    
def ploting_wave(sound):    
    length=len(sound)
    i=1
    plt.figure()
    for freq in sound:        
        plt.subplot(length,1,i)
        librosa.display.waveplot(freq , sr=22050)    
        i += 1
    plt.show()
     
# Function for spectograms        
def plot_specgram(sound):
    i = 1
    length =len(sound)
    plt.figure()
    for f in sound:
        plt.subplot(length,1,i)
        specgram(f, Fs=22050)
        i += 1
    plt.show()    

# Function for logspectograms
def plot_logspecgram(sound):
    i = 1
    length = len(sound)
    plt.figure()
    for f in sound:
        plt.subplot(length,1,i)
        D = librosa.logamplitude(np.abs(librosa.stft(f))**2, ref_power=np.max)
        librosa.display.specshow(D,x_axis='time' ,y_axis='log')
        i += 1    
    plt.show() 


# Feature extraction with librosa
# Different sound features we do not include them in the final experiment
# These features are better for sound classification not for auto encoders    
def extract_feature(sound):
    stft = np.abs(librosa.stft(sound))
    mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=60).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(sound, sr=sr,n_mels=256).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sr).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


# Concatenate all names , features and labels for each wav file
# We do not use this function for the final results    
def concatenate(sound , groundTruth):
    features, labels , names = np.empty((0,341)), np.empty(0) , np.empty(0)
    print("\nConcatenating: names, features and labels for each wav file....")
    pbar = tqdm(total=len(sound)) # Specify the progressBar    
    for fn,name,lab in zip(sound,groundTruth[:,0],groundTruth[:,1]):        
        mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
        ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        features = np.vstack([features,ext_features])
        labels = np.append(labels, lab)
        names = np.append(names , name)
        pbar.update()
    pbar.close()    
    return np.array(names , dtype =str), np.array(features), np.array(labels, dtype = np.int)


def PLOTTING():
    # Plot the five first waves
    print("\n")
    flag1 = input("Visulize some of the WAVE_Plots? [y/n]        >_ ")
    if flag1 == 'Y' or flag1 == 'y' or flag1 == 'yes':    
        f1 = input("\nSpecify how many plots to visualize: (number)  >_")
        print("\nWave_Plots")
        ploting_wave(raw[0:int(f1)])    
        print("\n")
    
    # Plot the five first spectograms
    flag2 = input("Visualize some of the spectograms? [y/n]      >_ ")
    if flag2 == 'Y' or flag2 == 'y' or flag2 == 'yes':
        f2 = input("\nSpecify how many plots to visualize: (number)  >_")
        print("Spectograms")
        plot_specgram(raw[0:int(f2)])    
        print("\n")
    
    flag3 = input("Visualize some of the LogSpectograms? [y/n]   >_  ")
    if flag3 == 'Y' or flag3 == 'y' or flag3 == 'yes':
        f3=input("\nSpecify how many plots to visualize: (number)  >_")
    # Plot the first five logspectograms
        print("\nLogSpectograms")
        plot_logspecgram(raw[0:int(f3)])


def splitting():
    # Splitting dataset into train and test | 70%/30% respectively
    random.seed(1)
    all_ind = np.where(truth[:,0])[0]
    # Make train set contain 70% random observations of data
    tr_ind = random.sample(list(all_ind) , round(70/100 * len(raw)))
    train_sound=  raw[tr_ind]    
    # Make test set contain the rest 30%
    test_sound = raw[np.delete(all_ind , tr_ind)]
    
    return train_sound, test_sound


# updatable plot
# Call-back plot we omited because of duranium    
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []        
        self.fig = plt.figure()        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1        
        clear_output(wait=True)
        
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

# Call the class        
plot_losses = PlotLosses()



def train():
    # Ask for the batchsize
    batch = input("Provide a number for batch_size: ")
    #
    # Ask for epochs
    ep = input("\nProvide the number of epochs: ")
    # this is the size of our encoded representations
    input_img = Input(shape=(train_sound.shape[1],))
    ##
    encoded = Dense(1000,activity_regularizer=regularizers.l1(10e-3), activation='sigmoid')(input_img)
    #encoded = Dense(512, activation='relu')(encoded)
    encoded = Dense(256, activation='sigmoid')(encoded)
    ##
    decoded = Dense(1000, activation='sigmoid')(encoded)
    #decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(train_sound.shape[1], activation='tanh')(decoded)
    ##   
    autoencoder = Model(input_img, decoded)
    ##
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
    ##
    history = autoencoder.fit(train_sound, train_sound,
                    epochs=int(ep),
                    batch_size=int(batch),
                    shuffle=True,
                    validation_data=(test_sound, test_sound))
                    #, callbacks=[plot_losses])       
    ##                
    encoded_imgs = autoencoder.predict(test_sound)
    return encoded_imgs, batch, history.history['loss'], history.history['val_loss']


def viz_wav(x):        
    n = 5  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        
        # display original
        ax = plt.subplot(2, n, i + 1)
        librosa.display.waveplot(test_sound[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        librosa.display.waveplot(x[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    

# Function for plotting the losses    
def Loss(loss, val_loss):
    plt.figure(1)
    plt.plot(loss, label="Train_loss")
    plt.plot(val_loss, label="Test_loss")
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.title("Epoch Vs Losses")
    plt.legend()
    
    plt.show()   


# Run the script
if __name__ == '__main__':
    
    # Sounds
    raw, truth, sr = Loading()
    PLOTTING()
    train_sound, test_sound = splitting()
    
    # Network
    encoded, batch, loss, val_loss  = train()
    viz_wav(encoded)
    Loss(loss,val_loss)
    
    
    
    
    


















