import tensorflow as tf
from train import generate_and_save_images
from ConFig.configreader import ConfigReader
cfg = ConfigReader()
tf.train.latest_checkpoint()
model_generator = tf.model.load_model('')
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, cfg.noise_dim])
generate_and_save_images(model_generator, cfg.TotalEpoch, seed)