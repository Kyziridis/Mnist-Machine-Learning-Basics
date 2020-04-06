
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

flag = input("Permutation ? [y/n] : ")
if flag == "y":
    
    # Rabndom permutation
    perm = np.arange(x_train.shape[1])
    np.random.shuffle(perm)
    x_train = x_train[:,perm]
    x_test = x_test[:,perm]


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def NN():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='mean_squared_error',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
        # Probability matrix before classification
    y_prob = model.predict(x_test)
    test_labels = np.argmax(y_test, axis=1)
    # classification results
    y_pred = model.predict_classes(x_test)
    target_names= list(map(str, np.arange(10)))
    print(classification_report(np.argmax(y_test,axis=1) , y_pred,target_names=target_names))
    print("\t\tConfusion_Matrix: \n\n", confusion_matrix(np.argmax(y_test,axis=1) , y_pred))


    # Missclassified images
    miss_img = np.where(y_pred != test_labels)[0]
    
    # Softmax output for missxlassified images
    y_miss = y_prob[miss_img]
    
    y_miss_labels = np.argmax(y_miss,axis=1)
    
    y_true_labels = test_labels[miss_img]
    
    miss_error = np.empty(0, dtype=np.float128)
    for i,j in enumerate(zip(y_true_labels , y_miss_labels)):  
        
        #print( np.abs(np.subtract(y_miss[i][j[0]], y_miss[i][j[1]] , dtype=np.float128) ))
        miss_error = np.append(miss_error , np.abs(np.subtract(y_miss[i][j[0]], y_miss[i][j[1]] , dtype=np.float128) ))
    
    top = miss_error.argsort()[-3:][::-1]
    print("\nMiss_Indexes: " + str(miss_img[top]))
    print("\nTrue_Labels: " + str(y_true_labels[top]))
    print("\nMiss_Lables: " + str( y_miss_labels[top]))
    
    a,b,c = top
    la = [miss_error[a] , miss_error[b] , miss_error[c]]
    return miss_img[top], la
    

times = 1
index = []
er = []
for i in range(times):
    x1,x2 = NN()
    index.append(x1)
    er.append(x2)
    print("Time: %s"%i)


#np.save("cnn_mse",index)

    
