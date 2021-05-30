#libraries 
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras.models import Sequential 
import pandas as pd
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras import backend as K 
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D



# for plotting images (optional)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# getting data
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_fruit = os.path.join(train_dir, 'fruit')
train_person = os.path.join(train_dir, 'person')
train_cat = os.path.join(train_dir, 'cat')
validation_fruit = os.path.join(validation_dir, 'fruit')
validation_person = os.path.join(validation_dir, 'person')
validation_cat = os.path.join(validation_dir, 'cat')

num_fruit_tr = len(os.listdir(train_fruit))
num_person_tr = len(os.listdir(train_person))
num_cat_tr = len(os.listdir(train_cat))
num_fruit_val = len(os.listdir(validation_fruit))
num_person_val = len(os.listdir(validation_person))
num_cat_val = len(os.listdir(validation_cat))

total_train = num_fruit_tr + num_person_tr + num_cat_tr
total_val = num_fruit_val + num_person_val + num_cat_val

BATCH_SIZE = 32
IMG_SHAPE = 300 # square image
EPOCHS = 20
ACC=float(input ("accuracy(0 => base on EPOCHS ):"))



#generators

#prevent memorization
train_image_generator = ImageDataGenerator(
    rescale=1./255,
    
    )

validation_image_generator = ImageDataGenerator(
    rescale=1./255)


train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=validation_dir,
                                                           shuffle=False,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')




#showing one sample of training images
xtbatches , ytbatches = next(train_data_gen)
for i in range (0,3):
    image1 = xtbatches[i]
    plt.imshow(image1)
    plt.show()

#showing one sample of validation images
xvbatches , yvbatches = next(val_data_gen )
for i in range (0,3):
    image1 = xvbatches[i]
    plt.imshow(image1)
    plt.show()

#callback
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0.001,
                                         patience=1, mode="min", baseline=ACC)

#model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)), # RGB
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.5), # 1/2 of neurons will be turned off randomly
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(3, activation='softmax') #[0,0, 1] or [0,1, 0] or [1,0,0]

    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
if  ACC == 0 :
  history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
    )
else: 
  history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE))),
    callbacks=[callback]
    )
model.save("rps.h5")

# analysis
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = input("Please enter runed epochs :") # ask from user , dipends on ACC 
epochs_range = range(int(epochs))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#Running the Model
# img_path = 'C:/Users/parsa/Desktop/deep/Ptoject_files/input/person/person_0.jpg'
# img = load_img(img_path, target_size=(IMG_SHAPE, IMG_SHAPE))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# images = np.vstack([x])
# classes = model.predict(images, batch_size=10)

# print(classes)
# if classes[0][0]>0.5:
#     print("your image is a cat ")
# if classes[0][1]>0.5:
#     print("your image is a fruit ")
# if classes[0][2]>0.5:
#     print("your image is a person ")
# else:
#     print("non") 
    
