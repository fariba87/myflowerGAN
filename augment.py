# for illustration purpose
#Augment the data to include random horizontal flipping and resize the images to have size 64-by-64.

'''You will learn how to apply data augmentation in two ways:

1) Use the Keras preprocessing layers, such as tf.keras.layers.Resizing, tf.keras.layers.Rescaling, tf.keras.layers.RandomFlip, and tf.keras.layers.RandomRotation.
2) Use the tf.image methods, such as tf.image.flip_left_right, tf.image.rgb_to_grayscale, tf.image.adjust_brightness, tf.image.central_crop, and tf.image.stateless_random*.'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE
from utils import process_path
train_ds = tf.data.Dataset.list_files('D:/Afagh/flowers/flower_photos/*/*', shuffle=True)
train_ds = train_ds.map(process_path())
IMG_SIZE= 28
# as keras layer ###############################################
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])

model = tf.keras.Sequential([ # Add the augment layers you created earlier.
  data_augmentation,
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  # Rest of your model.
])

# with tf.image  ###############################################
def augment(image_label, seed):
  image, label = image_label
  image = tf.image.resize(image, (64,64))
  image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)  # Make a new seed.
  new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
  # Random crop back to the original size.
  image = tf.image.stateless_random_crop(
      image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
  # Random brightness.
  image = tf.image.stateless_random_brightness(
      image, max_delta=0.5, seed=new_seed)
  image = tf.clip_by_value(image, 0, 1)
  return image, label
train_ds = (
    train_ds
    .shuffle(1000)
    .map(augment, num_parallel_calls=AUTOTUNE)
    .batch(batch_size)
    .prefetch(AUTOTUNE)
)
#2

