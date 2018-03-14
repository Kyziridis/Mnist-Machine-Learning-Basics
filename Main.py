#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:49:30 2018

@author: toul
"""

import collections
import numpy as np
from itertools import combinations,  product
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt




# Import the data
train_In = np.loadtxt("data/train_in.csv", delimiter = ',')
train_Out = np.loadtxt("data/train_out.csv", delimiter = ',')
test_In = np.loadtxt("data/test_in.csv", delimiter = ',')
test_Out= np.loadtxt("data/test_out.csv", delimiter = ',')

# Preprocess the data#
######################
# Constuct a dictionary to group the data according to each class
d = {}
for i in range(len(train_Out)):
    if train_Out[i] not in d:
        d[train_Out[i]] = list()    
    d[train_Out[i]].append(train_In[i])        
# We have the dictionary of Cloud points for each digit from the trainset
# od dictionary is for the trainset    
# Sort the dictionary according to the keys
# [0,1,2,3,4,5,6,7,8,9]    
od = collections.OrderedDict(sorted(d.items()))

# Do the same for the test_set
d_test = {}
for i in range(len(test_Out)):
    if test_Out[i] not in d_test:
        d_test[test_Out[i]] = list()
    d_test[test_Out[i]].append(test_In[i])
# Sort     
od_test = collections.OrderedDict(sorted(d_test.items()))
##########################################################


"""Task1"""
print("---TASK_1---")
print("Calculate the centers of each image and then find similar images based on their centers-distance.")

# Calculate the centers(means) for each class
centers = list(map(lambda x: np.mean(x , axis=0), od.values()))

# Define the Map_Func for euclidian distance
def Map_Func(x,val):
    yy = list(map(lambda y: np.linalg.norm(x-y) , val)    )
    return np.max(yy)

# Calculate the radius for each class using the above function
rad = list( map(Map_Func , centers , od.values()) )

# NUmber of images in each class
num = list(map(len,od.values()))

# Construct a matrix include tha radius and the number of images for each class
print("\nNum of images in each class: " , num)
print("\nRadius for each class: " , np.round(rad,2))

# Calculate the distance of centers between all possible combinations of classes
dist = {} 
for c1, c2 in list(combinations(range(10),2)):
    dist[(c1,c2)] = np.linalg.norm(centers[c1] - centers[c2])

# Find the classes with minimum distance and maximum distance
x_min,y_min=list( dist.items())[  np.where(list(dist.values()) == np.min(list(dist.values()))  )[0][0] ]
x_max,y_max=list( dist.items())[  np.where(list(dist.values()) == np.max(list(dist.values()))  )[0][0] ]

# Question about the distance matrix
flag0 = input("\nPrint the distance matrix for all images?  [y/n] : ")
if flag0 == "y" or flag0 == "Y" or flag0 == "yes":
    list(map(print , dist.items()))[0]

print("\n  Conclusions ")
print("--------------------------------------------------------------")    
print("Most similar numbers:  " + str(x_min) + " with distance: " + str(y_min))
print("Most disimilar numbers:" + str(x_max) + " with distance: " + str(y_max)) 
print("--------------------------------------------------------------")
print("\n")


"""Task2"""
    
flag11 = input("Move to task_2?  [y/n] : ")
if flag11 == 'y' or flag11 == 'Y' or flag11 == 'yes':
    
    print("\n---TASK_2---")
    print("Calculate different distances between the train/test set and the centers.\nImplement the K-NN algorithm and measure the accuracy for the two datasets.")
    print("\n")
    
    # Calculate Euclidian distance using np.linalg.norm
    print("Accuracy using Euclidian distance.\n")
    
    # Train_set: Euclidian distance using numpy
    temp_euc = []
    for c1,c2 in list(product(range(1707),range(10))):
        temp_euc.append(np.linalg.norm(train_In[c1] - centers[c2]))    
    # Classify according to minimum distance
    clust_train = list(map(lambda x: np.where(x == np.min(x))[0][0] ,np.array(temp_euc).reshape((len(train_In),10)) ))
    print("Accuracy in train_set: %s" % np.round(len(np.where(clust_train == train_Out)[0])/len(train_Out),3) )
    
    # Test_set: Euclidian distance using numpy    
    temp=[]
    for c1,c2 in list(product(range(len(test_In)),range(10))):
        temp.append(np.linalg.norm(test_In[c1] - centers[c2]))
    # Classify according to minimum distance    
    clust_test = list(map(lambda x: np.where(x == np.min(x))[0][0] ,np.array(temp).reshape((len(test_In),10)) ))
    print("Accuracy in test_set: %s" %(len(np.where(clust_test == test_Out)[0])/len(test_Out)))
    
    # Confusion Matrices ##################################
    print("\nConfusion Matrix for Train_set")
    print(confusion_matrix(np.array(clust_train), train_Out))
    print("\nConfusion Matrix for Test_set")
    print(confusion_matrix(np.array(clust_test), test_Out))
    print("\n\n")
    ########################################################
    
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


"""Task3"""
flag22 = input("\nMove to task_3?  [y/n] : ")
if flag22 == 'y' or flag22 == 'Y' or flag22 == 'yes':
    
    print("\n---TASK_3---")
    print("Implement the Naiv-Bayes classifier for classes: 5 & 7.\nCalculate prior, likelihood and posterior probabilities.\nClassify the new image to that class with bigger posterior probability.")
        
    # Feature X: count the black pixels of all images for class 5 and 7
    x_5 = list(map(lambda x: len(np.where(x != -1)[0]) , od[5]))
    x_7 = list(map(lambda x: len(np.where(x != -1)[0]) , od[7]))
    
    # Plot hists for the above feature in order to see if it is helpfull on segmentation
    flag = input("\nPrint histogram for Num_of_Black_pixels feature?  [y/n] : ")
    if flag == "y" or flag == "Y" or flag=="yes":
        plt.hist(x_5 , histtype='step' , color='red' , label="Class: 5")
        plt.hist(x_7 , histtype='step' , label="Class: 7")
        plt.legend(loc=0)
        plt.title("Histogram for Num_Black_Pixels feature")
        plt.show()
    
    # Sum for class5 and class7
    SUM = num[5] + num[7]
    prior_5 = num[5]/SUM # Prior probability of P(C1)
    prior_7 = num[7]/SUM # Prior probability of P(C2)
    
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
    
    # Train_set: NaivByes
    nb = list(map(NaivBayes , od[5]+od[7]))
    Accuracy_train = (len(np.where(np.asarray(nb) == xx[1])[0])/len(xx[1]) )*100
    print("Accuracy in train_set:  %s " %np.round(Accuracy_train,3) )
    
    # Test_set: NaivByes
    # Construct dummies only for classes 5 and 7
    test_labels = list(np.repeat(5,len(od_test[5]))) + list(np.repeat(7,len(od_test[7])))
    nb_test = list(map(NaivBayes , od_test[5]+od_test[7]))    
    Accuracy_test = (len(np.where(np.asarray(nb_test) == test_labels)[0])/len(test_labels) ) *100
    print("Accuracy in test_set:  %s " %np.round(Accuracy_test,3) )
    


"""Task4"""
flag33 = input("\nMove to task_4?  [y/n] : ")
if flag33 == 'y' or flag33 == 'Y' or flag33 == 'yes':
    
    np.random.seed(666)
    def Perceptron(x , label, hta=0.01):
        # Generate random weights
        np.random.seed(666)
        # Construct the bias
        ones = np.ones(len(x))
        x = np.hstack((x , ones.reshape(-1,1))) 
        W = np.random.randn(x.shape[1] , 10)
        z = x@W
        
        # Calculate the hot vector of prediction 
        pred_norm = np.zeros_like(z)
        pred_norm[np.arange(len(x)) , np.argmax(z,axis=1)]=1
        
        # Construct the hot vector of groundTruth
        groundTruth = np.zeros((len(x),10))
        groundTruth[np.arange(len(x)) , label.astype(np.int64)] = 1
        
        # Calculate the error matrix
        err = groundTruth - pred_norm
        
        # Initialize the first values for the training loop
        dx = err.T@x 
        w_new = W + hta*dx.T
        
        # just a counter
        i = 1
        acc_l=[]
        # Start the training procedure
        while -1 in err:
            # Update the weights
            dx = err.T@x    
            w_new = w_new + hta*dx.T
            z = x@w_new
            
            # Calculate the hot vector of prediction 
            pred_norm = np.zeros_like(z)
            pred_norm[np.arange(len(x)) , np.argmax(z,axis=1)]=1
            
            # Calculate the error matrix
            err = groundTruth - pred_norm
                    
            # Measure the Accuracy in each Epoch
            y = list(map(lambda x: np.where(x == np.max(x))[0][0],z))
            acc = np.round(len(np.where(y == label)[0])/len(label),3)
            acc_l.append(acc)
            print("Epoch: %s "%i + " Accuracy: " + str(acc) + "")
            i += 1
        # Plot the Accuracy vs Error    
        plt.figure()    
        plt.title("Accuracy Vs Error")
        plt.xlabel("Iterations")
        plt.ylabel("Values")
        plt.plot(1-np.asarray(acc_l) , color= 'red' , label="Error")
        plt.plot(acc_l , label="Accuracy")
        plt.legend(loc=0)
        plt.show()    
        return y , acc ,w_new
    
    print("\n---TASK_4---")
    print("Implement a single-layer multiclass perceptron for image classification.\n")
    flag44 = input("Start training Perceptron, choose learning rate :")
    
    # Call the Perceptron function for the trainset
    y , accur, weights_trained = Perceptron(train_In , train_Out,hta=float(flag44) )  
    
    # Multiply the trained weights with the test_set
    test_wth_bias = np.hstack((test_In , np.ones(len(test_In)).reshape(-1,1)))
    mat = test_wth_bias@weights_trained
    classification = list(map(np.argmax , mat))
    
    # Measure Test_set accuracy
    Test_acc = np.round(len(np.where(classification == test_Out)[0])/len(test_Out),3)
    print("---------------------------")    
    print("Accuracy on train_set: %s"%accur)     
    print("Accuracy on test_set:  %s"%Test_acc)



    

"""Task5"""

def sigmoid(x):
    return 1/ (1+np.exp(-x)) 

def tan(x):
    return  (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))
    
def xor_net(x1,x2,w):
    # Write it analytical like the linear math equation
    # The two hidden nodes
    z1 = x1*w[0] + x2*w[1] + w[2]
    z2 = x1*w[3] + x2*w[4] + w[5]
    # Pluge them into sigmoid function
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)
    # Third node for the output
    z3 = a1*w[6] + a2*w[7] + w[8]
    # FINAL-Output for sigmoid
    out = sigmoid(z3)
    out = np.array(out)   
    return out

def mse(weights):
    target = np.array([0.,1.,1.,0.])
    out1 = xor_net(0.,0.,weights)
    out2 = xor_net(0.,1.,weights)
    out3 = xor_net(1.,0., weights)
    out4 = xor_net(1.,1., weights)
    out = np.array([out1, out2, out3, out4])
    mse = np.square(np.subtract(target, out)).mean()
    out[out>=0.5] = 1
    out[out!=1] = 0
    miss = len(np.where(out != target)[0])
    return np.array(mse) , miss
       
def grdmse(weights):
    epsilon = 0.001
    W_new = []
    for i in range(len(weights)):
        temp = np.copy(weights)
        temp[i] = temp[i] + epsilon
        Wg = temp
        W_new.append( (mse(Wg)[0] - mse(weights)[0])/epsilon  )
    return np.array(W_new)        

def testing():
    np.random.seed(666)
    #weights = np.random.normal(size=9).round(3)
    weights = np.random.uniform(0, 1, 9)
    hta = input("Provide the learning-rate value: ")
    mse_plot = []
    i = 1
    while mse(weights)[1] != 0:    
        weights = weights - float(hta)*grdmse(weights)
        yy , _ = mse(weights)
        mse_plot.append(yy)    
        print("Epoch: %s  "%i + "Error: " + str(mse(weights)[0]) + " Missclassified: " + str(mse(weights)[1]))
        i += 1
    plt.figure()
    plt.plot(mse_plot)
    plt.show()



# Call the above functions
flagg_teliko = input("\nMove to task_5?  [y/n] : ")
if flagg_teliko == "Y" or flagg_teliko == "y" or flagg_teliko == "yes":
    print("\n---TASK_5---")
    print("\nBuild a multyclass perceptron algorithm for the Xor problem using Gradient-Descent.")
    
    flag55 = input("\nStart training the Xor-Network?  [y/n] : ")
    if flag55 == 'y' or flag55 == 'Y' or flag55 == 'yes':
        testing()











        
        
        
