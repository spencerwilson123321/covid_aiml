from tensorflow.keras.preprocessing import image
import cv2
from tensorflow import keras
import numpy as np
from skimage.transform import resize

# # To capture video from web-cam
cap = cv2.VideoCapture(0)

classifier = keras.models.load_model('./mymodel')
classes = ['Correct mask', 'Improper mask', 'Without mask']

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    test_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to 128x128. We dont want to normalize so set preserve_range=True
    test_image = resize(test_image, (128, 128), preserve_range=True)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    # classifier.predict returns an array of probabilities. Pick the highest probable class
    result = np.argmax(classifier.predict(test_image), axis=-1)

    # Print the class label
    print(classes[result[0]])

    # Display video
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
