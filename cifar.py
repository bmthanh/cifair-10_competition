# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:40:16 2019

@author: thanh.bui
"""

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.optimizers import RMSprop, Adam, SGD
import pandas as pd
import numpy as np

#%% Load data
df=pd.read_csv("./data/trainLabels.csv")
# Append .png to id column
df['id'] = df['id'].apply(lambda x: str(x) + '.png')

def split_train_valid(data, valid_ratio=0.1):
    '''split data into training and validation sets with specified ratio
    '''
    np.random.seed(42) # To generate the same shuffled indices
    shuffled_indices = np.random.permutation(len(data))
    valid_set_size = int(len(data)*valid_ratio)
    train_indices = shuffled_indices[valid_set_size:]
    valid_indices = shuffled_indices[:valid_set_size]
    return data.iloc[train_indices], data.iloc[valid_indices]

# Split the data into training and validation sets
valid_ratio = 0.1
train_df, valid_df = split_train_valid(df, valid_ratio)
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True) 

datagen=ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(dataframe=train_df, 
                                            directory="./data/train", x_col="id", y_col="label", 
                                            class_mode="categorical", target_size=(32,32), batch_size=32)
valid_generator = datagen.flow_from_dataframe(dataframe=valid_df, 
                                            directory="./data/train", x_col="id", y_col="label", 
                                            class_mode="categorical", target_size=(32,32), batch_size=32)



#%% Construct and evaluate the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])



model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.samples // valid_generator.batch_size,
                    epochs=10)