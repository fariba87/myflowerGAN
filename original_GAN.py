# for illustration purpose (GAN which include only Dense layer
#
import tensorflow as tf
tf.random.set_seed(1)
codings_size = 30
# original GAN is composed of Dense Layer
generator = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(150, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(150, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(1, activation="sigmoid")])
gan = tf.keras.Sequential([generator, discriminator])
# at first discriminator is trained and then the generator(while fixing discriminator)
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")


batch_size = 32

'''def train_gan(gan, dataset, batch_size, codings_size, n_epochs):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])  # create noise
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([tf.cast(generated_images,tf.float32), X_batch[0][:,:,:,0]], axis=0)  # concate fake and real (one batch)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)  # label for fake =0 and real =1
            discriminator.train_on_batch(X_fake_and_real, y1)  # train discriminator on it
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])   # create noise
            y2 = tf.constant([[1.]] * batch_size)  # label for fake=1
            gan.train_on_batch(noise, y2)  # train gan
train_gan(gan, dataset, batch_size, codings_size, n_epochs=50)'''

'''That’s it! After training, you can randomly sample some codings from a
Gaussian distribution, and feed them to the generator to produce new images:'''
codings = tf.random.normal(shape=[batch_size, codings_size])
generated_images = generator.predict(codings)  # prediction using saved generator