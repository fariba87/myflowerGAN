import tensorflow as tf
from ConFig.configreader import ConfigReader
cfg = ConfigReader()
#from main import class_names


# def get_label(file_path):
#   #Convert the path to a list of path components
#   parts = tf.strings.split(file_path, os.path.sep)
#   # The second to last is the class-directory
#   one_hot = parts[-2] == class_names
#   # Integer encode the label
#   return tf.argmax(one_hot)


def decode_img(img):
  img = tf.io.decode_jpeg(img, channels=3)
  return tf.image.resize(img, [cfg.img_height, cfg.img_width])


def process_path(file_path):
  # label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img # label  # if i dont need the label just return img
