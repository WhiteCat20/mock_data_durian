import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

img = image.load_img("train/ripe/ripe_heatmap_ripe_durian_sensor_data_1.png")
plt.imshow(img)

cv2.imread("train/ripe/ripe_heatmap_ripe_durian_sensor_data_1.png").shape

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('train/', target_size=(200,200), batch_size=3, class_mode='binary')
test_dataset = train.flow_from_directory('test/', target_size=(200,200), batch_size=3, class_mode='binary')

model = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200,200,3)),
 tf.keras.layers.MaxPool2D(2,2), 
 tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
 tf.keras.layers.MaxPool2D(2,2), 
 tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
 tf.keras.layers.MaxPool2D(2,2), 
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(512, activation='relu'),
 tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])

model_fit = model.fit(train_dataset, steps_per_epoch=3, epochs=30, validation_data=test_dataset) 

dir_val = 'validation'
model_input_shape = (200, 200)

for i in os.listdir(dir_val):
    img_path = os.path.join(dir_val, i)
    
    # Load and resize the image
    img = image.load_img(img_path, target_size=model_input_shape)
    plt.imshow(img)
    plt.show()
    
    # Convert the image to a numpy array
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)
    
    # Make predictions
    val = model.predict(X)
    
    # Print the result
    if val[0][0] == 0:
        print('ripe')
    else:
        print('unripe')