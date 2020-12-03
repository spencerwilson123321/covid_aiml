from tensorflow.keras.preprocessing import image
import cv2
from tensorflow import keras
import numpy as np
from skimage.transform import resize

# # To capture video from web-cam
cap = cv2.VideoCapture(0)

classifier = keras.models.load_model('./mymodel')

# THESE MAY BE IN THE WRONG ORDER
classes = ['Correct mask', 'Improper mask', 'Without mask']

while True:
    # Read the frame
    _, img = cap.read()
    test_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    test_image = resize(test_image, (128, 128), preserve_range=True)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = np.argmax(classifier.predict(test_image), axis=-1)
    print(classes[result[0]])

    # # Display
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
