import tensorflow as tf
from Modules.mycallbacks import checkpoint
from model import Discriminator, Generator
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from IPython import display
# from main import train_ds
from loss import generator_loss, discriminator_loss
from ConFig.configreader import ConfigReader
# from tensorflow.keras.utils import plot_model
cfg = ConfigReader()
acc = tf.keras.metrics.Accuracy()
from read_tfdata import train_ds
logs = {}
import datetime
log_dir ="logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tf_call=tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
file_writer = tf.summary.create_file_writer(log_dir)
_callbacks = [ tf_call, checkpoint]
callbacks = tf.keras.callbacks.CallbackList(_callbacks, add_history=True)  # ????
# model_G = Generator()
# model_G()
#
# model_generator = model_G.model
# model_D = Discriminator()
# model_D()
# model_discriminator = model_D.model
def load_Generator_Discriminator_model():
    model_G = Generator()
    model_G()
    model_generator = model_G.model
    model_D = Discriminator()
    model_D()
    model_discriminator = model_D.model
    return model_G, model_D, model_generator, model_discriminator
model_G, model_D, model_generator, model_discriminator = load_Generator_Discriminator_model()
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, cfg.noise_dim])

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=model_G.optimizer,  # generator_optimizer,
                                 discriminator_optimizer=model_D.optimizer,  # discriminator_optimizer,
                                 generator=model_generator,  # generator,
                                 discriminator=model_discriminator)  # discriminator)


# acc = tf.keras.metrics.Accuracy() [FID score for GAN performance]


def generate_and_save_images(model, epoch1, test_input):

    predictions = model(test_input, training=False)

    # print(predictions[0].shape)
    img = tf.keras.preprocessing.image.array_to_img((predictions[0]*127.)+127.5)#predictions[0])
    print(type(img))

    img.save(f"fake_images/img{epoch}_{epoch1}.png")
    with file_writer.as_default():
        tf.summary.image('generated image', predictions , step=0)


def train_data_for_one_epoch(images,epoch):

    noise = tf.random.normal([cfg.batchSize, cfg.noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = model_generator(noise, training=True)
        real_output = model_discriminator(images, training=True)
        # numObservations = tf.shape(real_output)[0]
        # idx = np.random.rand(numObservations) < flipProb;
        # A = real_output.numpy()
        # A[idx]=1-A[idx]
        # A = tf.convert_to_tensor(A)
        #  real_output(idx) = 1 - reaYReal(:,:,:, idx);
        fake_output = model_discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output) # A


        with file_writer.as_default():
            tf.summary.scalar('G loss', gen_loss, step=epoch)
            tf.summary.scalar('D loss', disc_loss, step=epoch)
        #

    gradients_of_generator = gen_tape.gradient(gen_loss, model_generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model_discriminator.trainable_variables)

    model_G.optimizer.apply_gradients(zip(gradients_of_generator, model_generator.trainable_variables))
    model_D.optimizer.apply_gradients(zip(gradients_of_discriminator, model_discriminator.trainable_variables))

flipProb = 0.35;

std=0.1
for epoch in range(cfg.TotalEpoch):
    print('epoch', epoch+1)
    start = time.time()
   # callbacks.on_epoch_begin(epoch, logs=logs)

    for image_batch in train_ds:
        if epoch%100 ==0 & epoch!=0:
            std =std*0.01
            image_batch = image_batch+ tf.random.normal(shape=tf.shape(image_batch),
                             mean=0.0,
                             stddev = std,#(0.01)*(i+1),#/epoch+1,
                             dtype=tf.float32)

        train_data_for_one_epoch(image_batch,epoch)
    if epoch%50 == 0:
        generate_and_save_images(model_generator, epoch, seed)



    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    model_generator

# Generate after the final epoch
display.clear_output(wait=True)
num_examples_to_generate = 1
seed = tf.random.normal([num_examples_to_generate, cfg.noise_dim])
generate_and_save_images(model_generator, cfg.TotalEpoch, seed)

# tensorboard --logdir logs/fit