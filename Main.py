#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:49:30 2018

@author: toul
"""

import collections
import numpy as np
from itertools import combinations, permutations, product
from tqdm import tqdm
from functools import reduce
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.plotly as py
import pylab

# Import the data
train_In = np.loadtxt("/home/dead/Documents/NN/data/train_in.csv", delimiter = ',')
train_Out = np.loadtxt("/home/dead/Documents/NN/data/train_out.csv", delimiter = ',')

test_In = np.loadtxt("/home/dead/Documents/NN/data/test_in.csv", delimiter = ',')
test_Out= np.loadtxt("/home/dead/Documents/NN/data/test_out.csv", delimiter = ',')


"""Task1"""
print("---TASK_1---")
print("Calculate the centers of each image and then find similar images based on their centers-distance.")

# Constuct a dictionary to group the data according to each class
d = {}
for i in range(len(train_Out)):
    if train_Out[i] not in d:
        d[train_Out[i]] = list()    
    d[train_Out[i]].append(train_In[i])    # We have the dictionary of Cloud points for each digit

# Sort the dictionary according to the keys
# [0,1,2,3,4,5,6,7,8,9]    
od = collections.OrderedDict(sorted(d.items()))

# Calculate the centers(means) for each class
centers = list(map(lambda x: np.mean(x , axis=0), od.values()))

# Define the Map_Func for euclidian distance
def Map_Func(x,val):
    yy = list(map(lambda y: np.linalg.norm(x-y) , val)    )
    return np.max(yy)

# Calculate the radius for each class
rad = list( map(Map_Func , centers , od.values()) )

# NUmber of images in each class
num = list(map(len,od.values()))

# Calculate the distance for all possible combinations of classes
dist = {} 
for c1, c2 in list(combinations(range(10),2)):
  dist[(c1,c2)] = np.linalg.norm(centers[c1] - centers[c2])

x0,y0=list( dist.items())[  np.where(list(dist.values()) == np.min(list(dist.values()))  )[0][0] ]
x1,y1=list( dist.items())[  np.where(list(dist.values()) == np.max(list(dist.values()))  )[0][0] ]

# Question about the distance matrix
flag0 = input("\nPrint the distance matrix for all images?  [y/n] : ")
if flag0 == "y" or flag0 == "Y" or flag0 == "yes":
    list(map(print , dist.items()))[0]

print("\n  Conclusions ")
print("--------------------------------------------------------------")    
print("Most similar numbers:  " + str(x0) + " with distance: " + str(y0))
print("Most disimilar numbers:" + str(x1) + " with distance: " + str(y1) ) 
print("--------------------------------------------------------------")
print("\n")



"""Task2"""

print("---TASK_2---")
print("Calculate different distances between the train/test set and the centers.\nImplement the K-NN algorithm and measure the accuracy for the two datasets.")
print("\n")

# Calculate Euclidian distance using np.linalg.norm
print("Accuracy using Euclidian distance.\n")

temp_euc = []
for c1,c2 in list(product(range(1707),range(10))):
    temp_euc.append(np.linalg.norm(train_In[c1] - centers[c2]))

clust_train = list(map(lambda x: np.where(x == np.min(x))[0][0] ,np.array(temp_euc).reshape((len(train_In),10)) ))
print("Accuracy in train_set: %s" %(len(np.where(clust_train == train_Out)[0])/len(train_Out)))

temp=[]
for c1,c2 in list(product(range(len(test_In)),range(10))):
    temp.append(np.linalg.norm(test_In[c1] - centers[c2]))
    
clust_test = list(map(lambda x: np.where(x == np.min(x))[0][0] ,np.array(temp).reshape((len(test_In),10)) ))
print("Accuracy in test_set: %s" %(len(np.where(clust_test == test_Out)[0])/len(test_Out)))

print("\nConfusion Matrix for Train_set")
print(confusion_matrix(np.array(clust_train), train_Out))
print("\nConfusion Matrix for Test_set")
print(confusion_matrix(np.array(clust_test), test_Out))
print("\n\n")

# Calculate the distances for Train set    
clust_euc = list(map(lambda x: np.where(x == np.min(x))[0][0] ,pairwise_distances(train_In , centers ) ))
clust_cos = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(train_In , centers , metric='cosine')))
clust_mahal = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(train_In , centers , metric='mahalanobis')))
clust_sqeucl = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(train_In , centers , metric='sqeuclidean')))
clust_city = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(train_In , centers , metric='cityblock')))

# Calculate the distances for Test set    
clust_euc_test = list(map(lambda x: np.where(x == np.min(x))[0][0] ,pairwise_distances(test_In , centers ) ))
clust_cos_test = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(test_In , centers , metric='cosine')))
clust_mahal_test = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(test_In , centers , metric='mahalanobis')))
clust_sqeucl_test = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(test_In, centers , metric='sqeuclidean')))
clust_city_test = list(map(lambda x: np.where(x == np.min(x))[0][0] , pairwise_distances(test_In, centers , metric='cityblock')))

# Printing Accuracies
print("Accuracy       TRAIN_SET | TEST_SET")
print("----------------------------------")
print("Euclidean_dist:  | %s" % (np.round(len(np.where(clust_euc == train_Out)[0])/len(train_Out),3))+" | %s |"%(np.round(len(np.where(clust_euc_test == test_Out)[0])/len(test_Out),3)))
print("Cosine_dist:     | %s" % (np.round(len(np.where(clust_cos == train_Out)[0])/len(train_Out),3))+" | %s |"%(np.round(len(np.where(clust_cos_test == test_Out)[0])/len(test_Out),3)))
print("Mahalanobis_dist:| %s" % (np.round(len(np.where(clust_mahal == train_Out)[0])/len(train_Out),3))+" | %s |"%(np.round(len(np.where(clust_mahal_test == test_Out)[0])/len(test_Out),3)))
print("Sqeuclidean_dist:| %s" % (np.round(len(np.where(clust_sqeucl == train_Out)[0])/len(train_Out),3))+" | %s |"%(np.round(len(np.where(clust_sqeucl_test == test_Out)[0])/len(test_Out),3)))
print("Cityblock_dist:  | %s" % (np.round(len(np.where(clust_city == train_Out)[0])/len(train_Out),3))+" | %s |"%(np.round(len(np.where(clust_city_test == test_Out)[0])/len(test_Out),3)))
print("----------------------------------")
print("\n")
############################################################

"""Task3"""
print("\n---TASK_3---")
print("Implement the Naiv-Bayes classifier for classes: 5 & 7.\nCalculate prior, likelihood and posterior probabilities.\nClassify the new image to that class with bigger posterior probability.")

# Calculate number of black pixels_feature for all the images
num_of_blacks=[]
for i in range(10):
    num_of_blacks.append( sum(list(map(lambda x: len(np.where(x != -1)[0]) , od[i])) ) )

# Feature X: count the black pixels of all images for class 5 and 7
x_5 = list(map(lambda x: len(np.where(x != -1)[0]) , od[5]))
x_7 = list(map(lambda x: len(np.where(x != -1)[0]) , od[7]))

# Plot hists for the above feature in order to see if it is helping on segmentation
flag = input("\nPrint histogram for Num_of_Black_pixels feature?  [y/n] : ")
if flag == "y" or flag == "Y" or flag=="yes":
    plt.hist(x_5 , histtype='step' , color='red' , label="Class: 5")
    plt.hist(x_7 , histtype='step' , label="Class: 7")
    plt.legend(loc=0)
    plt.title("Histogram for Num_Black_Pixels feature")
    plt.show()

# Sum for class5 and class7
SUM = 88+166
prior_5 = 88/SUM # Prior probability of P(C1)
prior_7 = 166/SUM # Prior probability of P(C2)

# Create object xx contains the feature and groundtruth
# xx[0]: feature of class 5 & 7
# xx[1]: label groundTruth
x5 = np.array([x_5, np.repeat(5,len(x_5))])
x7 = np.array([x_7,np.repeat(7,len(x_7))])
xx = np.hstack((x5,x7))

# Split the data into 10-bins
bining = np.linspace(np.min(xx[0]) , np.max(xx[0]),10)

# Calculate the frequency of each bin for each class
freq_5 = np.histogram(x_5 , bining)[0]
freq_7 = np.histogram(x_7 , bining)[0]

# Calculate liklihood and the posterior for each class
def NaivBayes(x):
    """ 
    Function for NaivBayes classifier.
        
    Input: Array
    Output: The label(5 or 7) according to the posterior 
        
    When a new image comes:
    1. Measure its feature
    2. Place it into the corresponding bin
    3. Calculate the posterior for the two classes
    4. Return the Class for the bigger posterior """
    feature = len(np.where(x != -1)[0]) 
    BIN = np.digitize(feature,bining , right=True ) 
    post_5 = ( freq_5[BIN-1]/sum(freq_5) )*prior_5
    post_7 = (freq_7[BIN-1]/sum(freq_7))*prior_7
    if post_5 > post_7:
        #print("Classified in: 5" )
        classified = 5
        return classified
    else:
        #print("Classified in: 7")
        classified = 7
        return classified

# Form the NaivBayes classifier into all images for 5 and 7
nb = list(map(NaivBayes , od[5]+od[7]))
Accuracy = (len(np.where(np.asarray(nb) == xx[1])[0])/len(xx[1]) )*100





# Plotting the two histograms for the variance feature
flag1 = input("\nPlot variance-feature histogram?  [y/n] :")
if flag1 == "y" or flag1=="Y" or flag1 == "yes":
    var_5 = list(map(np.var , od[5]))
    var_7 = list(map(np.var, od[7]))
    pylab.figure(figsize=(15,10))
    pylab.hist(var_5 , histtype='step' , label='var class: 5',color='red')
    pylab.hist(var_7 , histtype='step' , label='var class: 7',color='blue')
    pylab.legend(loc=0)
    pylab.title("Variance histograms for classes 5,7")
    pylab.savefig('Histogram for variances.png')
    pylab.show()




























        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
