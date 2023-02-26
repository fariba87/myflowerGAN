# FlowerGAN
This tensorflow project is based on deep convolutional GAN(DCGAN) model  and tf-flower dataset in tfds(tensorflow dataset) for generating flower images.

As an ML project, we should consider two main objects:
    
1) Data: 
2) Model


### Load Training Data    
  	1) Download and extract the Flowers data set 
	2) Create an image datastore containing the photos of the flowers.
  	3) Augment the data to include random horizontal flipping and resize the images to have size 64-by-64.
### Generative Adversarial Network:
A GAN is a framework that's used to generate new data by learning the distribution of data
The GAN framework is comprised of two neural networks, the generator, and
discriminator, 
In the context of image generation, the generator generates fake data, when given noise as
input, and the discriminator classifies real images from fake images. During training, the
generator and the discriminator compete with each other in a game and as a result, get
better at their jobs. The generator tries to generate better-looking images to fool the
discriminator and the discriminator tries to get better at identifying real images from fake
images.

GANs are still evolving and new applications emerge every day. Some of these applications
include artistic image generation, data augmentation, image-to-image translation, super
resolutions, and video synthesis
	- original GAN consists of Dense layers
	- as the model architecture(in help of matlab) is based on deep convolutional GAN, it uses conv layars(DCGAN)


## Deep Convolutional GAN (DCGAN)

### Data Transorfation 
1) normalize inputs to the range of [-1, 1]  
2) im_normalized  = (img-127.5)/127.5
### Generator
1) Converts the random vectors of size 100 to 4-by-4-by-512 arrays using a project layer and reshape operation.
2) Upscales the resulting arrays to 64-by-64-by-3 arrays using a series of transposed convolution layers with batch normalization and ReLU layers.
3) outout should be of size (BS,64,64,3)
4) use kernel initializer N(0,0.02)




Note: remember to resale output again when you generate data by generator (img*127 +127.5)

### Discriminator
1) similar to binary classification
2) based on convolutional layer
3) pooling layers are replaced with stridden convolution layers
4) outout should be of size (BS,1,1,1)
5) add noise to Discriminator input
6) use kernel initializer N(0,0.02)




### More Generative models
1) ProGAN (progressively generate high resolution image)
2) StyleGAN (based on proGAN with some modification )
3) SRGAN
4) diffusion models(easier to train -slower to run)
### Exploring the model
1) Number of parameters in G and D Net are of order of 5M
2) Number of flowers in tf-flower dataset is of order of 3000
3) We need data augmentation 
4) a stable GAN will have
    1. D loss around 0.5 (0.5-0.8)
    2. stability for D around epoch 100-300
    3. G loss> D loss (around 1.0,1.5,2.0 or even higher
5) if D loss converges to zero, it means we have strong D
    1. solution: adding noise to D input (i did this one)
    2. solution: impair D by randomly giving false labels to real images
    3. flip the Discriminator output for real image with some proportion (i wrote the code for that too) 

#### Tensorboard dev
to share the result of tensorboard with others (run td-dev.sh))
#### Tensorboard 
run :  tensorboard --logdir logs/fit
