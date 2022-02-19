# Model structure
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


# Paths

train_path = r"D:\Fire_project\Fire_project\fire_dataset\train"
valid_path = r"D:\Fire_project\Fire_project\fire_dataset\valid"
test_path = r"D:\Fire_project\Fire_project\fire_dataset\test"

# ImageDataGenerator (image preprocessing)

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(105,165), classes=['fire', 'non_fire'], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(105,165), classes=['fire', 'non_fire'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(105,165), classes=['fire', 'non_fire'], batch_size=10, shuffle=False)

# Model

model = Sequential()
#
model.add(Conv2D(32, (3, 3), activation="relu",
                 input_shape=(105, 165, 3)))
model.add(MaxPool2D((2, 2)))
#
model.add(Conv2D(64, (3, 3),
                 activation="relu", padding="same"))
model.add(Dropout(0.2))
model.add(MaxPool2D((2, 2)))
#
model.add(Conv2D(128, (3, 3),
                 activation="relu", padding="same"))
model.add(Dropout(0.5))
model.add(MaxPool2D((2, 2)))
#
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(2, activation="sigmoid"))

# Model compilation
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Summary of the model
print(model.summary())

# Model run

model.fit(train_batches, validation_data=valid_batches, epochs=20, verbose=1)

# Result

path_project = r"C:\Users\NorbertK\PycharmProjects\Fire_detection_project"

results = pd.DataFrame(model.history.history)
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(results['accuracy'], label='Training Accuracy', color='green', marker='*')
plt.plot(results['val_accuracy'], label='Training Accuracy', color='red', marker='*')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(results['loss'], label='Training Loss', color='green', marker='*')
plt.plot(results['val_loss'], label='Val Loss', color='red', marker='*')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path_project + r"\reports\figures\Loss_Accuracy.png")

# Save model

model.save(r"C:\Users\NorbertK\PycharmProjects\Fire_detection_project\src\model\model.h5")