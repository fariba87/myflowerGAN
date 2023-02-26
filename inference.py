import tensorflow as tf

model_generator = tf.model.load_model('')
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, cfg.noise_dim])
generate_and_save_images(model_generator, cfg.TotalEpoch, seed)