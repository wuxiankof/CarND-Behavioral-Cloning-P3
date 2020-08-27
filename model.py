import os
import csv
import cv2
import math
import numpy as np
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# folder and file names
folder_path = '../data/'
filename_csv = 'driving_log.csv'

# define generator
def generator(samples, batch_size=32):
    
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            # read batch samples
            images = []
            angles = []
            for batch_sample in batch_samples:
                
                # read steering measurement for the center camera image
                steering_center = float(batch_sample[3])

                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                # read in images from center, left and right cameras
                path = folder_path + 'IMG/'
                img_center = cv2.imread(path + batch_sample[0].split('/')[-1])
                img_left = cv2.imread(path + batch_sample[1].split('/')[-1])
                img_right = cv2.imread(path + batch_sample[2].split('/')[-1])

                # add images and angles to data set
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])
            
            # Data Augmentation by flipping images
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)
            
            # convert to numpy array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# read csv file
samples = []
with open(folder_path + filename_csv, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Set  batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

# model structure
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3))) # normalization
model.add(Cropping2D(cropping=((70,25),(0,0)))) # cropping

model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(Convolution2D(64, (3, 3), activation='relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, 
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
                    validation_data=validation_generator, 
                    validation_steps=math.ceil(len(validation_samples)/batch_size), 
                    epochs=5, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
