import tensorflow as tf
from dataset import ImagenetDataset, CifarDataset
from network import Resnet_Cifar, Resnet_Cifar_Tweak
from datetime import datetime
from utils import WarmupExponential, WarmupCosineLRDecay, mix_up, label_smoothing
import shutil
import os
import time
import math


# constant
BATCH_SIZE = 512
EPOCHS = 120


@tf.function
def train_network(x, y):
    with tf.GradientTape() as tape:
        outp = model(x, training=True)
        loss = loss_obj(y, outp)
        loss_l2 = loss + tf.math.add_n(model.losses)
    gradients = tape.gradient(loss_l2, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_mean.update_state(loss)
    train_l2_loss_mean.update_state(loss_l2)


@tf.function
def valid_network(x, y):
    outp = model(x, training=False)
    loss = loss_obj(y, outp)

    valid_loss_mean.update_state(loss)
    top_1_acc.update_state(y, outp)
    top_5_acc.update_state(y, outp)


# dataset
ds = CifarDataset('/tf_datasets', ds_name='cifar10')
train_size = ds.train_num
train_ds = ds.get_train_ds(BATCH_SIZE)
valid_ds = ds.get_valid_ds(BATCH_SIZE)


step_per_epoch = math.ceil(train_size / BATCH_SIZE)
# base
# model = Resnet_Cifar(category=10)
# lr = tf.keras.optimizers.schedules.ExponentialDecay(1e-1, math.ceil(train_size / BATCH_SIZE) * 30, .1, True)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=.9)

# efficient
# linear scaling & lr warmup & zero gamma & no bias decay
# model = Resnet_Cifar(category=10, bn_gamma='zeros', bias_regularizer=None)
# lr = WarmupExponential(2e-1, step_per_epoch * 5, step_per_epoch * 30, .1, True)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=.9)

# network tweak
# resnet-D
# model = Resnet_Cifar_Tweak(category=10, bn_gamma='zeros', bias_regularizer=None)
# lr = WarmupExponential(2e-1, step_per_epoch * 30, .1, True, step_per_epoch * 5)
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=.9)

# training refinements
# cosine lr / 120 epochs
model = Resnet_Cifar_Tweak(category=10, bn_gamma='zeros', bias_regularizer=None)
lr = WarmupCosineLRDecay(2e-1, step_per_epoch * EPOCHS, step_per_epoch * 10)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=.9)


model.build((None, 224, 224, 3))
model.summary()

# loss & metric
loss_obj = tf.keras.losses.CategoricalCrossentropy()

train_loss_mean = tf.keras.metrics.Mean(name='train_loss')
train_l2_loss_mean = tf.keras.metrics.Mean(name='train_l2_loss')
valid_loss_mean = tf.keras.metrics.Mean(name='valid_loss')
top_1_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_acc')
top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc')


# tensorboard
now = datetime.now()
log_dir = './tensorboard/training_refinement-1-{}'.format(now.strftime("%Y%m%d-%H%M%S"))
shutil.rmtree(os.path.join(log_dir, '*'), ignore_errors=True)

train_loss_log_dir = os.path.join(log_dir, 'train_loss')
train_l2_loss_log_dir = os.path.join(log_dir, 'train_l2_loss')
valid_loss_log_dir = os.path.join(log_dir, 'valid_loss')
top_1_log_dir = os.path.join(log_dir, 'top_1_acc')
top_5_log_dir = os.path.join(log_dir, 'top_5_acc')

train_loss_writer = tf.summary.create_file_writer(train_loss_log_dir)
train_l2_loss_writer = tf.summary.create_file_writer(train_l2_loss_log_dir)
valid_loss_writer = tf.summary.create_file_writer(valid_loss_log_dir)
top_1_writer = tf.summary.create_file_writer(top_1_log_dir)
top_5_writer = tf.summary.create_file_writer(top_5_log_dir)

# train
report_format = """\
{} epoch - t_loss : {:.3f} / t_l2_loss : {:.3f} / v_loss : {:.3f}
           top_1 : {:.3f} / top_5 :{:.3f} - time : {}"""

for epoch in range(EPOCHS):
    train_loss_mean.reset_states()
    train_l2_loss_mean.reset_states()
    valid_loss_mean.reset_states()
    top_1_acc.reset_states()
    top_5_acc.reset_states()

    t_start = time.time()
    for img, label in train_ds:
        # training refinement - start
        label_smooth = label_smoothing(label)
        img, label = mix_up(img, label_smooth)
        # training refinement - end
        train_network(img, label)
    with train_loss_writer.as_default():
        tf.summary.scalar('loss', train_loss_mean.result(), step=epoch)
    with train_l2_loss_writer.as_default():
        tf.summary.scalar('loss', train_l2_loss_mean.result(), step=epoch)

    for img, label in valid_ds:
        # training refinement - start
        label = label_smoothing(label)
        # training refinement - end
        valid_network(img, label)
    with valid_loss_writer.as_default():
        tf.summary.scalar('loss', valid_loss_mean.result(), step=epoch)
    with top_1_writer.as_default():
        tf.summary.scalar('acc', top_1_acc.result(), step=epoch)
    with top_5_writer.as_default():
        tf.summary.scalar('acc', top_5_acc.result(), step=epoch)
    t_end = time.time()

    print(report_format.format(epoch,
                               train_loss_mean.result(),
                               train_l2_loss_mean.result(),
                               valid_loss_mean.result(),
                               top_1_acc.result(),
                               top_5_acc.result(),
                               int(t_end - t_start)))
