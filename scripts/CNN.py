import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# validation_path = '..\\assets\\validation\\'

SAVE_MODEL = False

training_path = '..\\assets\\training\\'
testing_path ='..\\assets\\testing\\'


idg = ImageDataGenerator()

# validation_data = idg.flow_from_directory(validation_path, color_mode='grayscale',target_size=(128,128), shuffle=True)
# train_data = idg.flow_from_directory(training_path, color_mode='grayscale',target_size=(128,128))
# test_data = idg.flow_from_directory(testing_path, color_mode='grayscale',target_size=(128,128))

train_data = image_dataset_from_directory(
    training_path,
    labels='inferred',
    color_mode='grayscale',
    image_size=(128, 128),
    label_mode='categorical',
    shuffle=True,
    validation_split=0.1,
    subset='training',
    seed=1123
)

validation_data = image_dataset_from_directory(
    training_path,
    labels='inferred',
    color_mode='grayscale',
    image_size=(128, 128),
    label_mode='categorical',
    shuffle=True,
    validation_split=0.1,
    subset='validation',
    seed=1123
)

test_data = image_dataset_from_directory(
    testing_path,
    labels='inferred',
    color_mode='grayscale',
    image_size=(128, 128),
    label_mode='categorical',
    shuffle=False
)

num_classes = 3

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(3,3),activation='relu', input_shape=(128,128,1)))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=80,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(filters=100,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# Keras Metrics
metrics = tf.keras.metrics

model.compile(optimizer='adam',
              loss="categorical_crossentropy",
              metrics = [
                  'accuracy',
                  metrics.Precision(),
                  metrics.Recall()
              ]
              )

# fit the model to training data
history = model.fit(train_data, epochs=10, verbose=1)

if SAVE_MODEL:
    model.save('mymodel')

loss, acc, precision, recall = model.evaluate(test_data, verbose=1)
print('Test Set: Loss = %.3f || Accuracy = %.3f || Precision = %.3f || Recall = %.3f' % (loss, acc, precision, recall))


# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()