from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

training_path = '.\\assets\\training\\'
testing_path = '.\\assets\\testing\\'

train = ImageDataGenerator(rescale=1 / 255)
validation = ImageDataGenerator(rescale=1 / 255)

train_data = train.flow_from_directory(training_path)
test_data = train.flow_from_directory(testing_path)

num_classes = 3

model = Sequential()
model.add(Conv2D(filters=6, kernel_size=(3,3),activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(2,2))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu', strides=(1,1), padding="same"))
model.add(MaxPooling2D(2,2))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu', strides=(1,1), padding="same"))
model.add(MaxPooling2D(2,2))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu', strides=(1,1), padding="same"))
model.add(MaxPooling2D(2,2))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu',  strides=(1,1),padding="same"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss="categorical_crossentropy",
              metrics=['accuracy'])

# fit the model to training data
model.fit(train_data, epochs=10, verbose=1)

loss, acc = model.evaluate(test_data, verbose=1)
print('Test accuracy = %.3f' % acc)
