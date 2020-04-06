#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 19:31:51 2018

@author: dead
"""

'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import time
from IPython.display import clear_output
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import collections

batch_size = 158
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

flag = input("Permutation ? [y/n] : ")
if flag == "y":
    
    # Rabndom permutation
    perm = np.arange(x_train.shape[1])
    np.random.shuffle(perm)
    x_train = x_train[:,perm]
    x_test = x_test[:,perm]


# updatable plot
# This script can run in cpu so we keep the live-plot    
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

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def Model_Init():
        
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='linear'))
    
    #model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    return model

def Simulation():
    
    model = Model_Init()
    
    now = time.time()
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=1,
                        verbose=1,
                        callbacks=[plot_losses],
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    print("------------------------------------")
    print('Test loss:', np.round(score[0],2))
    print('Test accuracy:', np.round(score[1],2) )
    print("Time on CPU: " + str(time.time() - now),"sec")
    print("------------------------------------")
    
    # Probability matrix before classification
    y_prob = model.predict(x_test)
    test_labels = np.argmax(y_test, axis=1)
    # classification results
    y_pred = model.predict_classes(x_test)
    target_names= list(map(str, np.arange(10)))
    print(classification_report(np.argmax(y_test,axis=1) , y_pred,target_names=target_names))
    print("\t\tConfusion_Matrix: \n\n", confusion_matrix(np.argmax(y_test,axis=1) , y_pred))
                
    # Make dictionaries for grouped output for each class
    d_test = {}
    for i in range(len(test_labels)):
        if test_labels[i] not in d_test:
            d_test[test_labels[i]] = list()
        d_test[test_labels[i]].append(y_prob[i])
    # Sort     
    out = collections.OrderedDict(sorted(d_test.items()))
    
    # Find missclassified instances for each class
    miss_indx = []
    miss_labels = []
    d_miss = {}
    error = []
    for i in range(10):
        # missclassified instances [index]
        miss_indx.append(np.where(np.argmax(out[i],axis=1) != i)[0])
        # np.argmax on missclassified instances
        miss_labels.append(np.argmax(np.array(out[i])[miss_indx[i]],axis=1))
        # Collect the missclassified prob vectors for each digit
        d_miss[i]=list(np.array(out[i])[miss_indx[i]])
        # Take the mean vector of missclassified
        x = np.mean(d_miss[i] , axis=0)
        # np.argmax for the missclassified class
        y = np.argmax(x,axis=0)
        # Measure the error between the actual class probability and the missclassified one
        error.append(np.abs(x[i] - x[y]))
        print("Class: %s ---> %s  Error:  " %(i,y) + str(error[i]))

    return error, history       



# Run the NN 10 times 
sim_times = 1    
err_sim = {}
if __name__ == "__main__":
    
    for i in range(sim_times):
        print("\n\nSimulation runs ---> %s/%s"%(i,sim_times))
        simulation = Simulation()
        err_sim[i] = list(simulation)
    # Calculate the mean of the errors    
    mean_error = np.mean(list(err_sim.values()) , axis=0)
    print("Top-3 missclassified digits: " + str(mean_error.argsort()[-3:][::-1]))


