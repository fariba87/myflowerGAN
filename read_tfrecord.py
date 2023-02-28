import tensorflow as tf
import tensorflow_datasets as tfds
ds, metadata = tfds.load(
    'tf_flowers',
    #split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=False)  # no need for labels
# then there will a folder created under tensorflow_dataset directory consist of tf record files of images
path_tfrec='path to that folder'
tfrec_ds = tf.data.TFRecordDataset(path_tfrec)
feature_description = {
    'image': tf.io.FixedLenFeature([] , dtype=tf.string),
    #'label': tf.io.FixedLenFeature([] , dtype=tf.string)
}
def _parse_function(im):
    return tf.io.parse_single_example(im, feature_description)
parsed_dataset = tfrec_ds.map(_parse_function)



