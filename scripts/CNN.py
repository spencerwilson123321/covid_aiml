import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

training_path = '..\\assets\\training\\'
validation_path = '..\\assets\\validation\\'
testing_path ='..\\assets\\testing\\'


idg = ImageDataGenerator()


train_data = idg.flow_from_directory(training_path, color_mode='grayscale',target_size=(128,128))
validation_data = idg.flow_from_directory(validation_path, color_mode='grayscale',target_size=(128,128), shuffle=True)
test_data = idg.flow_from_directory(testing_path, color_mode='grayscale',target_size=(128,128))

num_classes = 3

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3),activation='relu', input_shape=(128,128,1)))
# model.add(MaxPooling2D(2,2))
# model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
# model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(filters=80,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
# model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(filters=100,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
# model.add(ZeroPadding2D(padding=(1,1)))
# model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
# model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss="categorical_crossentropy",
              metrics = ['accuracy'])


# fit the model to training data
history = model.fit(train_data, validation_data=validation_data, epochs=10, verbose = 1)
loss, acc = model.evaluate(test_data, verbose = 1)
print('Test accuracy = %.3f' % acc)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

