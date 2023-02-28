'''In the first phase, we train the discriminator. A batch of real images is
sampled from the training set and is completed with an equal number of
fake images produced by the generator. The labels are set to 0 for fake
images and 1 for real images, and the discriminator is trained on this
labeled batch for one step, using the binary cross-entropy loss.
Importantly, backpropagation only optimizes the weights of the
discriminator during this phase.
In the second phase, we train the generator. We first use it to produce
another batch of fake images, and once again the discriminator is used to
tell whether the images are fake or real. This time we do not add real
images in the batch, and all the labels are set to 1 (real): in other words,
we want the generator to produce images that the discriminator will
(wrongly) believe to be real! Crucially, the weights of the discriminator
are frozen during this step, so backpropagation only affects the weights
of the generator

The generator never actually sees any real images, yet it gradually learns to produce
convincing fake images! All it gets is the gradients flowing back through the
discriminator. Fortunately, the better the discriminator gets, the more information about
the real images is contained in these secondhand gradients, so the generator can make
significant progress
'''