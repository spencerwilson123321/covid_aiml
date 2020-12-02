from tensorflow.keras.preprocessing import image
# import cv2
import math
from random import random
from tensorflow import keras
import numpy as np


classifier = keras.models.load_model('./mymodel')
classes = ['Correct mask', 'Without mask', 'Incorrect mask']


def predict_class(test_image):
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    return classifier.predict(test_image)

print()
print("Masks Correct")
for i in range(0, 5):
    img = image.load_img(f'../assets/testing/masks_correct/{i}.jpg', color_mode="grayscale")
    result = predict_class(img)
    print(result[0])

print()
print("Masks Incorrect")
for i in range(0, 5):
    img = image.load_img(f'../assets/testing/masks_incorrect/{i}.jpg', color_mode="grayscale")
    result = predict_class(img)
    print(result)

print()
print("Without Masks")
for i in range(0, 5):
    img = image.load_img(f'../assets/testing/without_masks/{i}.jpg', color_mode="grayscale")
    result = predict_class(img)
    print(result)
