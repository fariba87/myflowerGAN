import tensorflow as tf
from tensorflow import keras
import tempfile
import os
MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version)) #''  # where you want to save model
tf.saved_model.simple_save(keras.backend.get_session(),
                           export_path,  #
                           inputs={'input_image':model.input},
                           outputs={t.name for t in model.output})