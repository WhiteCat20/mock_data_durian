import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import os 
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def choose_image_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    file_path = filedialog.askopenfilename(
        title="Choose an image file",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
    )

    return file_path

loaded_model = tf.keras.models.load_model('saved_model')

dir_val = 'validation'

# Allow the user to choose an image file using the file explorer
image_filename = choose_image_file()

# Check if the user canceled the file selection
if not image_filename:
    print("File selection canceled.")
else:
    # Specify the target size expected by the model
    model_input_shape = (200, 200)

    # Load and resize the image
    img_path = image_filename
    img = image.load_img(img_path, target_size=model_input_shape)
    plt.imshow(img)
    plt.show()

    # Convert the image to a numpy array
    X = image.img_to_array(img)
    X = np.expand_dims(X, axis=0)

    result = loaded_model.predict(X)

    # Print the result
    if result[0][0] == 0:
        print('ripe')
    else:
        print('unripe')