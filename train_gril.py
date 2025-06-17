import tensorflow as tf
import numpy as np
import random
import os

from batch_loader import generate_gril
from utils import total_samples_count
from models import gril
from losses import action_loss

batch_size = 32
train_datapath = "training_data/train"
val_datapath = "training_data/val"

file_list = os.listdir(train_datapath)
val_list = os.listdir(val_datapath) # Loads file names from the train/val folders.

random.shuffle(file_list) # Randomizes training data for better generalization.

total_train_samples = total_samples_count(train_datapath, file_list)
total_val_samples = total_samples_count(val_datapath, val_list)

train_steps_per_epoch = int(np.ceil(total_train_samples/batch_size))
val_steps_per_epoch = int(np.ceil(total_val_samples/batch_size))

# Uses .npz file counts to calculate total samples and batch steps for training/validation.

print("Converting Train dataset")
tfx = tf.data.Dataset.from_generator(
    generate_gril,
    args=[train_datapath, file_list],
    output_signature=(
        {
            "image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),#Input: image (3 channels), depth (1 channel).
            "depth": tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32)
        }, {
            "gaze": tf.TensorSpec(shape=(2,), dtype=tf.float32),#Output: gaze (x, y), action (4 control values).
            "action": tf.TensorSpec(shape=(4,), dtype=tf.float32)
        }
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

print("Converting Val dataset")
val = tf.data.Dataset.from_generator(
    generate_gril,
    args=[val_datapath, val_list],
    output_signature=(
        {
            "image": tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),#Input: image (3 channels), depth (1 channel).
            "depth": tf.TensorSpec(shape=(224, 224, 1), dtype=tf.float32)
        }, {
            "gaze": tf.TensorSpec(shape=(2,), dtype=tf.float32),#Output: gaze (x, y), action (4 control values).
            "action": tf.TensorSpec(shape=(4,), dtype=tf.float32)
        }
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# model training

model = gril()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.00001,
    decay_steps=10000,
    decay_rate=0.9
    #Uses Exponential Decay to reduce learning rate as training progresses.
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-4)

model.compile(
    optimizer=optimizer,
    loss=[action_loss, 'mean_squared_error'] #action_loss (custom weighted loss for control prediction),Second loss: MSE (default) for gaze prediction.
)

model.fit(
    tfx,
    epochs=30,
    validation_data=val,
    callbacks=[tf.keras.callbacks.CSVLogger('gril_training.log')]
    #Trains the model for 30 epochs, Validates on val, Logs training info into gril_training.log
)

model.save("gril.h5") #Saves the entire trained model (architecture + weights + optimizer state) to gril.h5
