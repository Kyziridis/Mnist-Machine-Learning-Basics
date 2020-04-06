import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.models import Model
from keras.layers import Input, Dense

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,BatchNormalization,Activation
from keras import backend as K

from keras.datasets import cifar10
(data1, y_train), (data_test, y_test) = cifar10.load_data()



"------------------------------------------------------ CONVOLUTIONAL AUTOENCODER"


data1 = data1.astype('float32') / 255
data_test = data_test.astype('float32') / 255
data1 = np.reshape(data1, (len(data1), 32, 32, 3)) 
data_test = np.reshape(data_test, (len(data_test), 32, 32, 3)) 
 

input_img = Input(shape=(32,32,3))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


history = autoencoder.fit(data1, data1,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(data_test, data_test))

predicted = autoencoder.predict(data_test)

#n = 10  # how many digits we will display
#plt.figure(figsize=(20, 4))
#for i in range(n):
    # display original
   # ax = plt.subplot(2, n, i + 1)
    #plt.imshow(data_test[i])
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

    # display reconstruction
    #ax = plt.subplot(2, n, i + 1 + n)
    #plt.imshow(predicted[i])
    #plt.gray()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
#plt.savefig('cifar_plt6.png')


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.savefig('losses.png')





















