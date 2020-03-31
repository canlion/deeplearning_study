import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pdb
import time
import datetime
import math
import argparse

from dataset import VOCDatasetYOLOv1
from loss import YOLOv1Loss
from utils import WarmupCosineLRDecay, LRWarmup
from network import YoloV1 as YoloV1


IN_MEAN = tf.constant([123.68, 116.779, 103.939], shape=(1, 1, 1, 3))
IN_STD = tf.constant([58.393, 57.12, 57.375], shape=(1, 1, 1, 3))


@tf.function
def normalization(img):
    # return img / 255.
    return (img-IN_MEAN)/IN_STD


@tf.function
def train_network(x, y):
    with tf.GradientTape() as tape:
        outp = model(normalization(x), training=True)
        loss = loss_obj(y, outp)
        loss_l2 = loss + tf.math.add_n(model.losses)
    gradients = tape.gradient(loss_l2, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_l2_loss.update_state(loss_l2)


# mixed precision
@tf.function
def train_network_mp(x, y):
    with tf.GradientTape() as tape:
        outp = model(tf.cast(normalization(x), tf.float16), training=True)
        loss = loss_obj(y, tf.cast(outp, tf.float32))
        loss_l2 = loss + tf.math.add_n(model.losses)
        scaled_loss = optimizer.get_scaled_loss(loss_l2)
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_l2_loss.update_state(loss_l2)


@tf.function
def valid_network(x, y):
    outp = model(normalization(x), training=False)
    loss = loss_obj(y, tf.cast(outp, tf.float32))

    valid_loss.update_state(loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', default=64, type=int, help='batch size.')
    # parser.add_argument('--init_lr', default=1e-3, type=float, help='initial learning rate.')
    parser.add_argument('--epochs', default=135, type=int, help='total training epochs.')
    parser.add_argument('--lr_list', default=[1e-4, 1e-5, 1e-6], help='lr list', nargs='*', type=float)
    parser.add_argument('--epochs_list', default=[75, 105], help='lr decay epochs.', nargs='*')
    parser.add_argument('--mode', choices=['normal', 'mixed_precision'], default='normal',
                        help='training mode, "normal" or "mixed_precision".')
    parser.add_argument('--warmup_start_lr', default=1e-5, type=float, help='init lr of warmup of learning rate.')
    parser.add_argument('--warmup_epochs', default=1, type=int, help='epochs of warmup of learning rate.')
    args = parser.parse_args()

    # dataset
    ds = VOCDatasetYOLOv1('/tf_datasets', 2007, 448, aug=True)
    train_ds, train_size = ds.dataset_train_batch(args.batch)
    valid_ds, valid_size = ds.dataset_valid_batch(args.batch)
    print('train data : {} / validation data : {}'.format(train_size, valid_size))

    # lr, optmizer
    step_per_epoch = math.ceil(train_size / args.batch)
    # lr = WarmupCosineLRDecay(args.init_lr, step_per_epoch * args.epochs, step_per_epoch * args.warmup_epochs)
    step_list = [step_per_epoch * i for i in args.epochs_list]
    print(args.lr_list, step_list)
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(step_list, args.lr_list)
    lr = LRWarmup(lr, args.warmup_epochs * step_per_epoch, args.warmup_start_lr)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=.9)
    # optimizer = tf.keras.optimizers.Adam(lr)

    # loss & metric
    loss_obj = YOLOv1Loss()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_l2_loss = tf.keras.metrics.Mean(name='train_l2_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')

    # set mixed precision
    if args.mode == 'mixed_precision':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('compute : {}, variables : {}'.format(policy.compute_dtype, policy.variable_dtype))
        tf.config.optimizer.set_jit(True)
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    # network
    model = YoloV1()
    model.build((args.batch, 448, 448, 3))
    model.load_weights('backbone_w/min_loss_model_w')
    model.summary()

    # tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_path = 'tensorboard/' + current_time + '/{}'
    train_log_dir = log_dir_path.format('train')
    train_l2_log_dir = log_dir_path.format('train_l2')
    valid_log_dir = log_dir_path.format('valid')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    train_l2_summary_writer = tf.summary.create_file_writer(train_l2_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)

    # train
    loss_min = None

    if args.mode == 'mixed_precision':
        train_func = train_network_mp
        print('mixed precision training')
    else:
        train_func = train_network

    report_format = '{} epoch - train loss : {:.3f} / validation loss : {:.3f} / time : {} / {}'
    for epoch in range(args.epochs):
        train_loss.reset_states()
        train_l2_loss.reset_states()
        valid_loss.reset_states()

        t_start = time.time()
        for img, label in train_ds:
            train_func(img, label)
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
        with train_l2_summary_writer.as_default():
            tf.summary.scalar('loss', train_l2_loss.result(), step=epoch)
        t_end = time.time()

        for img, label in valid_ds:
            valid_network(img, label)
        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=epoch)
        v_end = time.time()

        if (loss_min is None) or valid_loss.result() < loss_min:
            # model.save('yolo_save/yolo_min_loss', save_format='tf')
            model.save_weights('yolo_save/yolo_min_loss_w')
            loss_min = valid_loss.result()

        print(report_format.format(epoch, train_loss.result(), valid_loss.result(),
                                   int(t_end - t_start), int(v_end-t_end)))
