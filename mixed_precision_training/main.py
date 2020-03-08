import time
import math
import os
import argparse
from datetime import datetime

import tensorflow as tf

from network import Resnet50Half
from dataset import ImagenetDataset
from utils import WarmupCosineLRDecay


@tf.function
def train_network(x, y):
    with tf.GradientTape() as tape:
        outp = model(x, training=True)
        loss = loss_obj(y, tf.keras.activations.softmax(outp))
        loss_l2 = loss + tf.math.add_n(model.losses)
    gradients = tape.gradient(loss_l2, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_mean.update_state(loss)
    train_l2_loss_mean.update_state(loss_l2)


# mixed precision
@tf.function
def train_network_mp(x, y):
    with tf.GradientTape() as tape:
        outp = model(tf.cast(x, tf.float16), training=True)
        loss = loss_obj(y, tf.keras.activations.softmax(tf.cast(outp, tf.float32)))
        loss_l2 = loss + tf.math.add_n(model.losses)
        scaled_loss = optimizer.get_scaled_loss(loss_l2)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_mean.update_state(loss)
    train_l2_loss_mean.update_state(loss_l2)


@tf.function
def valid_network(x, y):
    outp = model(x, training=False)
    loss = loss_obj(y, tf.keras.activations.softmax(outp))

    valid_loss_mean.update_state(loss)
    top_1_acc.update_state(y, outp)
    top_5_acc.update_state(y, outp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=128, type=int, help='batch size.')
    parser.add_argument('--epochs', default=60, type=int, help='total training epochs.')
    parser.add_argument('--mode', choices=['normal', 'mixed_precision'], default='normal',
                        help='training mode, "normal" or "mixed_precision".')
    parser.add_argument('--init_lr', default=1e-1, type=float, help='initial learning rate.')
    parser.add_argument('--warmup_epochs', default=3, type=int, help='epochs of warmup of learning rate.')
    args = parser.parse_args()

    # dataset
    ds = ImagenetDataset('/tf_datasets', 'imagenet2012')
    train_size = ds.train_num
    train_ds = ds.get_train_ds(args.batch)
    valid_ds = ds.get_valid_ds(args.batch)

    step_per_epoch = math.ceil(train_size / args.batch)
    lr = WarmupCosineLRDecay(args.init_lr, step_per_epoch * args.epochs, step_per_epoch * args.warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=.9, nesterov=True)

    # loss & metric
    loss_obj = tf.keras.losses.CategoricalCrossentropy()

    train_loss_mean = tf.keras.metrics.Mean(name='train_loss')
    train_l2_loss_mean = tf.keras.metrics.Mean(name='train_l2_loss')
    valid_loss_mean = tf.keras.metrics.Mean(name='valid_loss')
    top_1_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='top_1_acc')
    top_5_acc = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_acc')

    # set mixed precision
    if args.mode == 'mixed_precision':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('compute : {}, variables : {}'.format(policy.compute_dtype, policy.variable_dtype))
        tf.config.optimizer.set_jit(True)
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    # network
    model = Resnet50Half()

    # tensorboard
    now = datetime.now()
    log_dir = './tensorboard/{}-{}'.format(args.mode, now.strftime("%Y%m%d-%H%M%S"))

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
               top_1 : {:.3f} / top_5 :{:.3f} - time : {} / {}"""

    if args.mode == 'mixed_precision':
        train_func = train_network_mp
        print('mixed precision training')
    else:
        train_func = train_network

    for epoch in range(args.epochs):
        train_loss_mean.reset_states()
        train_l2_loss_mean.reset_states()
        valid_loss_mean.reset_states()
        top_1_acc.reset_states()
        top_5_acc.reset_states()

        t_start = time.time()
        for img, label in train_ds:
            train_func(img, label)
        with train_loss_writer.as_default():
            tf.summary.scalar('loss', train_loss_mean.result(), step=epoch)
        with train_l2_loss_writer.as_default():
            tf.summary.scalar('loss', train_l2_loss_mean.result(), step=epoch)
        t_end = time.time()

        for img, label in valid_ds:
            valid_network(img, label)
        with valid_loss_writer.as_default():
            tf.summary.scalar('loss', valid_loss_mean.result(), step=epoch)
        with top_1_writer.as_default():
            tf.summary.scalar('acc', top_1_acc.result(), step=epoch)
        with top_5_writer.as_default():
            tf.summary.scalar('acc', top_5_acc.result(), step=epoch)
        v_end = time.time()

        print(report_format.format(epoch,
                                   train_loss_mean.result(),
                                   train_l2_loss_mean.result(),
                                   valid_loss_mean.result(),
                                   top_1_acc.result(),
                                   top_5_acc.result(),
                                   int(t_end - t_start),
                                   int(v_end - t_end)))
