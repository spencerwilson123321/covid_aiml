import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt


training_path = '..\\assets\\training\\'
testing_path ='..\\assets\\testing\\'


train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_data = train.flow_from_directory(training_path)
test_data = train.flow_from_directory(testing_path)

num_classes = 3

model = Sequential()
model.add(Conv2D(16,(3,3),activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss="categorical_crossentropy",
              metrics = ['accuracy'])

# fit the model to training data
model.fit(train_data, epochs=10, verbose = 1)

loss, acc = model.evaluate(test_data, verbose = 1)
print('Test accuracy = %.3f' % acc)