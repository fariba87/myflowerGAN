import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import decode_img, process_path#, get_label,
print('is GPU available:',tf.config.list_physical_devices('GPU'))
print('tf version:',tf.__version__)
from ConFig.configreader import ConfigReader
cfg = ConfigReader()

AUTOTUNE = tf.data.AUTOTUNE
# version 1  ##################################################
# train_ds = tf.keras.utils.image_dataset_from_directory('D:/Afagh/flowers/flower_photos/',
#                                                        validation_split=0.2,
#                                                        subset="training",
#                                                        seed=123,
#                                                        image_size=(cfg.img_height, cfg.img_width),
#                                                        batch_size=cfg.batchSize)
# version 2  ##################################################
# train_ds, metadata = tfds.load(
#     'tf_flowers',
#     #split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
#     with_info=True,
#     as_supervised=True,
# )
# version 3  ##################################################
# import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# path_to_downloaded_file = tf.keras.utils.get_file(
#     "flower_photos",
#     dataset_url,
#     untar=True)
# version 4  ##################################################
list_ds = tf.data.Dataset.list_files('D:/Afagh/flowers/flower_photos/*/*', shuffle=True) #false
train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# if seperate for validation:#################################
# image_count = len(list_ds)
# val_size = int(image_count * 0.2)
# train_ds = list_ds #.skip(val_size)
# val_ds = list_ds.take(val_size)


explore_data = False
if explore_data:
  plt.figure(figsize=(10, 10))
  for images, labels in train_ds.take(1):
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
    plt.show()
  #
  for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break
  for im in list_ds.take(2):
    img = tf.io.read_file(im)
    img = decode_img(img)
    img
    plt.imshow(img)#.numpy().astype("uint8"))
    #plt.title(class_names[labels[i]])
    plt.axis("off")
    print(img.shape)
    plt.show()
  for image_batch in list_ds.take(2):
    print(image_batch)



def augment(x):
  x = tf.image.resize(x, (cfg.img_height, cfg.img_width))
  # tf.cast(x, tf.float32)
  x = (x-127.5)/127.5
  # x = tf.image.random_flip_left_right(x)
  # x = tf.image.random_flip_up_down(x)
  # x = tf.image.rot90(x)
  # x = tf.image.random_brightness(x,0.2)
  return x

train_ds = train_ds.map(lambda x: (augment(x)))#.(num_parallel_calls=AUTOTUNE)
train_ds.shuffle(1000)
train_ds = train_ds.shuffle(1000).batch(cfg.batchSize,drop_remainder=True).prefetch(buffer_size=AUTOTUNE)
print(len(train_ds))
#val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(cfg.batchSize) # it is needed for evaluation

