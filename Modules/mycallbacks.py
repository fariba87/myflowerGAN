import os
import tensorflow as tf


root_logdir=  "./dataflower/my_logs_GAN"# os.path.join(os.curdir, "../Data/my_logs_ctc")
def get_run_logdir(root_logdir):
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir(root_logdir= root_logdir)
#tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
tbCallBack=tf.keras.callbacks.TensorBoard(log_dir='..\my_logs_GAN', histogram_freq=0,  write_graph=True, write_images=True)
def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1**(epoch / s)
    return exponential_decay_fn

earlystopping = tf.keras.callbacks.EarlyStopping(patience=10)
exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor='loss',
                                    factor=0.5,
                                    patience=5,
                                    verbose=0,
                                    mode='auto',
                                    min_delta=0.2,
                                    cooldown=0,
                                    min_lr=0)



CHECKPOINT_DIR = "./data/" + 'ckpt' + "/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

#os.makedirs(os.path.join(CHECKPOINT_DIR, "bestModel"), exist_ok=True)
#filepath1 = os.path.join(os.curdir, "saved_model")
filepath2 = '/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5'
filepath = os.path.join(CHECKPOINT_DIR , filepath2)
backup_dir = filepath_ctc = os.path.join(CHECKPOINT_DIR ,'backup')
backup_ckpt = tf.keras.callbacks.BackupAndRestore(backup_dir=backup_dir)
checkpoint =tf.keras.callbacks.ModelCheckpoint(filepath,
                                              verbose=1,
                                              save_best_only=True, monitor ="loss")







