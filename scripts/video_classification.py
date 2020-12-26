from keras_preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow import keras
import numpy as np
from skimage.transform import resize


classifier = keras.models.load_model('./mymodel')
classes = ['Correct mask', 'Improper mask', 'Without mask']

# To capture video from web-cam
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

cascade_file_src = "./haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascade_file_src)

# 0: Correct mask
# 1: improper mask
# 2: without mask
# Color order is BGR
rect_color={0:(0,128,0),1:(0,69,255), 2:(0,0,255)}

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    test_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(test_image, 1.2, 5)

    for (x, y, w, h) in faces:

        # Get the area of the face
        face_img = test_image[y:y + h, x:x + w]

        # Resize to 128x128. We dont want to normalize so set preserve_range=True
        resized_face = resize(face_img, (128, 128), preserve_range=True)

        face = image.img_to_array(resized_face)
        face = np.expand_dims(face, axis=0)

       # classifier.predict returns an array of probabilities. Pick the highest probable class
        prediction = classifier.predict(face)
        result = np.argmax(prediction, axis=-1)
        percentage = prediction[0][result][0] * 100

        # put rectangle around area of the face
        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color[result[0]], 2)

        # Print the class label and percentage
        print(classes[result[0]] + ' ' + str(percentage))

        # putting the text in the live feed
        cv2.putText(img, classes[result[0]] + ' ' + str(percentage), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color[result[0]], 2)

    # Resize to 128x128. We dont want to normalize so set preserve_range=True
    # test_image = resize(test_image, (128, 128), preserve_range=True)
    # test_image = image.img_to_array(test_image)
    # test_image = np.expand_dims(test_image, axis=0)
    #
    # # classifier.predict returns an array of probabilities. Pick the highest probable class
    # prediction = classifier.predict(test_image)
    # result = np.argmax(prediction, axis=-1)
    # percentage = prediction[0][result][0] * 100
    #
    # # Print the class label and percentage
    # print(classes[result[0]], percentage)

    # Display video
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()