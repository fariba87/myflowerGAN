import tensorflow as tf
import math
from ConFig.configreader import ConfigReader
cfg = ConfigReader()
tf.random.set_seed(1)

#tf.keras.initializers.glorot_normal
def getweights(gain=math.sqrt(2)):
    return tf.keras.initializers.VarianceScaling(gain)

kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.03)
#kernel_initializer = getweights(0.2)


def des_block(kernel_size, num_filter, stride, padding):
    return tf.keras.Sequential([tf.keras.layers.Conv2D(filters=num_filter,
                                                       kernel_size=kernel_size,
                                                       strides=stride,
                                                       padding=padding,
                                                       kernel_initializer=kernel_initializer),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.LeakyReLU(cfg.leakyreluscale)])


def gen_block(kernel_size, num_filter, stride, padding):
    return tf.keras.models.Sequential([tf.keras.layers.Conv2DTranspose(filters=num_filter,
                                                                       kernel_size=kernel_size,
                                                                       strides=stride,
                                                                       padding=padding,
                                                                       kernel_initializer= kernel_initializer),
                                       tf.keras.layers.BatchNormalization(),
                                       tf.keras.layers.ReLU()])


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.input1 = tf.keras.layers.Input((100,))
        self.project = tf.keras.layers.Dense(4 * 4 * 512)
        self.reshape = tf.keras.layers.Reshape([4, 4, 512])
        self.final_conv = tf.keras.layers.Conv2DTranspose(filters=3,
                                                          kernel_size=5,
                                                          strides=2,
                                                          padding="same",
                                                          activation="tanh",
                                                          kernel_initializer=kernel_initializer)
        self.optimizer = tf.keras.optimizers.Adam(cfg.Adam_lr, cfg.Adam_gradientdecay, cfg.Adam_square_gd)
        self.model = None

    def __call__(self):
        y = self.project(self.input1)
        y = self.reshape(y)
        for i in range(3):
            y = gen_block(num_filter=cfg.scale_filter_G[i]*cfg.num_filter,
                          kernel_size=5,
                          stride=2,
                          padding="same",
                         )(y)
        y = self.final_conv(y)
        model = tf.keras.Model(inputs=self.input1,
                               outputs=y)
        self.model = model
        self.model.summary()
        return self


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.input1 = tf.keras.layers.Input(shape=(64, 64, 3))
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.conv1 = tf.keras.Sequential([tf.keras.layers.Conv2D(filters=cfg.num_filter,
                                                                 kernel_size=cfg.kernel_size,
                                                                 strides=2,
                                                                 padding='same',
                                                                 kernel_initializer=kernel_initializer),
                                          tf.keras.layers.LeakyReLU(cfg.leakyreluscale)])
        self.final_conv = tf.keras.layers.Conv2D(filters=1,
                                                 kernel_size=4,
                                                 strides=1,
                                                 padding="valid",
                                                 activation='sigmoid',
                                                 kernel_initializer=kernel_initializer)
        self.optimizer = tf.keras.optimizers.Adam(2e-4, 0.5, 0.999)
        self.model = None

    def __call__(self):
        y = self.dropout(self.input1)
        # y = self.dropout(self.input1 + tf.random.normal(shape=tf.shape(self.input1),
        #                                                 mean=0.0,
        #                                                 stddev=0.1,
        #                                                 dtype=tf.float32))
        y = self.conv1(y)
        for i in range(3): #3
            y = des_block(kernel_size=5,
                          num_filter=cfg.scale_filter_D[i]*cfg.num_filter,
                          stride=cfg.stride_D,
                          padding=cfg.padding)(y)
        y = self.final_conv(y)
        model = tf.keras.Model(inputs=self.input1,
                               outputs=y)
        self.model = model
        self.model.summary()
        return self
# G = Generator()
# model_G = Generator()
# model_G()
#
# model_generator = model_G.model
# model_D = Discriminator()
# model_D()
# model_discriminator = model_D.model
# # G()#.build(input_shape=(100,))
# noise = tf.random.normal((4,64,64,3))
# print(noise.shape)
# #print(model_generator(noise))
# print(model_discriminator(noise).shape)

# # G.summary()
# D = Discriminator()
# D().build(input_shape=(64,64,3))
# # D.summary()