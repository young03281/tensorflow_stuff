from urllib.request import urlretrieve
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL.Image as Image
import os
import keras
from keras import layers, models

data_dir = './data/'

print(tf.__version__)

for dir in os.listdir(os.path.join(data_dir)):
    for image in os.listdir(os.path.join(data_dir + '\\' + dir)):
        image = Image.open(data_dir + '\\' + dir + '\\' + image)
        if image.height == 256 and image.width == 256 : continue
        new_image = image.resize((256, 256))
        filename = image.filename
        image.close()
        new_image.save(filename)

img_height = 256

img_width = 256

batch_size = 32

classes_name = os.listdir(os.path.join(data_dir))

train_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# sparete as image and label
image_batch, labels_batch = next(iter(normalized_ds))
# check the first image max and min value
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print("minumum value:",np.min(first_image), "maximum values: ",np.max(first_image))

num_classes = len(classes_name)

inputs = keras.Input(shape=(img_height, img_width, 3))

data_augmentation = keras.Sequential(
  [
    inputs,
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# build model layers
model = models.Sequential([
    inputs,
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dense(256, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dense(256, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.005),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=40
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

#get train accuracy in history
acc = history.history['accuracy']
#get validation accuracy
val_acc = history.history['val_accuracy']
# get train loss 
loss = history.history['loss']
#get validation loss
val_loss = history.history['val_loss']

epochs_range = range(epochs)
# plot accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
# plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('result.png')

url = "data:image/jpeg;base64,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"

filename = "aaa.jpg"

urlretrieve(url, filename)

#load image
img = keras.utils.load_img(
    filename, target_size=(256, 256)
)
# image to array
img_array = keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

#make prediction
predictions = model.predict(img_array)

#get score
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(classes_name[np.argmax(score)], 100 * np.max(score))
)
model.save("yoga_model.keras")