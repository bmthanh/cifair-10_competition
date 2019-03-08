# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:40:16 2019

@author: thanh.bui
"""

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.python.keras.optimizers import RMSprop, Adam
import pandas as pd

df=pd.read_csv("./data/trainLabels.csv")
# Append .png to id column
df['id'] = df['id'].apply(lambda x: str(x) + '.png')
NO_TRAINING = 40000
train_df = df.iloc[:NO_TRAINING, :]
valid_df = df.iloc[NO_TRAINING:, :]
valid_df.reset_index(drop=True, inplace=True) 
datagen=ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_dataframe(dataframe=train_df, 
                                            directory="./data/train", x_col="id", y_col="label", 
                                            class_mode="categorical", target_size=(32,32), batch_size=32)
valid_generator = datagen.flow_from_dataframe(dataframe=valid_df, 
                                            directory="./data/train", x_col="id", y_col="label", 
                                            class_mode="categorical", target_size=(32,32), batch_size=32)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])



model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_generator.samples // train_generator.batch_size,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.samples // valid_generator.batch_size,
                    epochs=10)