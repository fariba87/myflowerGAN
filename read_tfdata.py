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
list_ds = tf.data.Dataset.list_files('../*/*', shuffle=True)  # false
#list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
data_dir = '../flower_photos/'
#image_count = len(class_names)
# print(len(list_ds))
# print(list_ds[0])

# val_size = int(image_count * 0.2)
train_ds = list_ds #.skip(val_size)
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

normalization_layer = tf.keras.layers.Rescaling(1/255.)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))



data_dir ='C:/Users/scc/Downloads/flower_photos/flower_photos/'
image_count = len(class_names)
#list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = tf.data.Dataset.list_files('D:/Afagh/flowers/flower_photos/*/*', shuffle=True) #false
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
#print(len(list_ds))
#print(list_ds[0])

val_size = int(image_count * 0.2)
train_ds = list_ds#.skip(val_size)
#val_ds = list_ds.take(val_size)

# print(tf.data.experimental.cardinality(train_ds).numpy())
train_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)


def augment(x):
  x = tf.image.resize(x, (cfg.img_height, cfg.img_width))
  tf.cast(x, tf.float32)
 # x = (x/255.)-1.
  # x = tf.image.random_flip_left_right(x)
  # x = tf.image.random_flip_up_down(x)
  # x = tf.image.rot90(x)
  # x = tf.image.random_brightness(x,0.2)



  return x
# resize_and_rescale = tf.keras.Sequential([
#   layers.Resizing(cfg.img_height, cfg.img_width),
#   layers.Rescaling(1./255)
# ])
#
# def prepare(ds, shuffle=False, augment=False):
#   # Resize and rescale all datasets.
train_ds = train_ds.map(lambda x: (augment(x)))#.(num_parallel_calls=AUTOTUNE)
train_ds =train_ds.map(lambda x:(x-127.5)/127.5)
train_ds.shuffle(1000)
train_ds = train_ds.batch(cfg.batchSize,drop_remainder=True)
train_ds.shuffle(1000)
train_ds = train_ds.shuffle(1000).batch(cfg.batchSize).prefetch(buffer_size=AUTOTUNE)
print(len(train_ds))
#val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE).batch(cfg.batchSize) # it is needed for evaluation

# trainAug = tf.keras.Sequential([
# 	#tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255),
# 	tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
# 	tf.keras.layers.experimental.preprocessing.RandomZoom(
# 		height_factor=(-0.05, -0.15),
# 		width_factor=(-0.05, -0.15)),
# 	tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)
# ])
# train_ds = (
# 	train_ds
# 	.shuffle(cfg.batchSize * 100)
# 	.batch(cfg.batchSize)
# 	.map(lambda x: trainAug(x),
# 		 num_parallel_calls=tf.data.AUTOTUNE)
# 	.prefetch(tf.data.AUTOTUNE)
# )
# train_ds = (
#     train_ds
#     .shuffle(1000)
#     .map(augment, num_parallel_calls=AUTOTUNE)
#     .batch(batch_size)
#     .prefetch(AUTOTUNE)
# )
#
# class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
# print(class_names)
#

# print(tf.data.experimental.cardinality(val_ds).numpy())
