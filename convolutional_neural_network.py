# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Check if TensorFlow is using GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.load_weights('model_weights.h5')
# cnn.fit(x = training_set, validation_data = test_set, epochs = 5)
# cnn.save_weights('model_weights.h5')

# Part 4 - Making some prediction
import numpy as np
from tensorflow.keras.preprocessing import image
#List of predictions in a folder :
import os
folder_path = "dataset/single_prediction"
Dictionary_of_files_and_predictions= {}
for filename in os.listdir(folder_path):
    # check if the file is a regular file (i.e., not a directory)
    if os.path.isfile(os.path.join(folder_path, filename)):
        test_image = image.load_img(f'dataset/single_prediction/{filename}', target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        training_set.class_indices
        if result[0][0] == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'
        Dictionary_of_files_and_predictions[str(filename[:filename.index('.')])]= prediction
for i,j in Dictionary_of_files_and_predictions.items():
    print(i,': ',j)

# Dictionary_of_files_and_predictions['g']