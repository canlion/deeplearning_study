import tensorflow as tf
import functools


REGULARIZER = tf.keras.regularizers.l2(5e-4)


class ResidualBlock_BN_D(tf.keras.Model):
    def __init__(self, filters, bn_gamma='ones', bn_beta='zeros',
                 weight_regularizer=REGULARIZER, bias_regularizer=None,
                 feature_halve=False, match_dims=False):
        """ResNet residual block. version-D"""
        super(ResidualBlock_BN_D, self).__init__()

        self.match_dims = match_dims
        self.feature_halve = feature_halve

        stride = 2 if feature_halve else 1
        self.conv_0 = tf.keras.layers.Conv2D(filters[0], 1, 1, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer)
        self.bn_0 = tf.keras.layers.BatchNormalization()

        self.conv_1 = tf.keras.layers.Conv2D(filters[1], 3, stride, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer)
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(filters[2], 1, 1, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer)
        self.bn_2 = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                       gamma_initializer=bn_gamma)

        if self.match_dims or self.feature_halve:
            if self.feature_halve:
                self.pool_sc = tf.keras.layers.AveragePooling2D(2, 2, 'same')
            self.conv_sc = tf.keras.layers.Conv2D(filters[2], 1, 1, 'same',
                                                  kernel_regularizer=weight_regularizer,
                                                  bias_regularizer=bias_regularizer)
            self.bn_sc = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv_0(input_tensor)
        x = self.bn_0(x, training=training)
        x = tf.keras.activations.relu(x, alpha=.1)

        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = tf.keras.activations.relu(x, alpha=.1)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)

        if self.match_dims or self.feature_halve:
            if self.feature_halve:
                shortcut = self.pool_sc(input_tensor)
            else:
                shortcut = input_tensor
            shortcut = self.conv_sc(shortcut)
            shortcut = self.bn_sc(shortcut, training=training)
        else:
            shortcut = input_tensor

        x = x + shortcut
        x = tf.keras.activations.relu(x, alpha=.1)

        return x


class YOLOv1(tf.keras.Model):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.resnet = tf.keras.applications.resnet.ResNet50(input_shape=(448, 448, 3),
                                                       include_top=False,
                                                       weights='imagenet')
        self.set_config()

        self.RB_0 = ResidualBlock_BN_D([512, 512, 512], feature_halve=True)

        self.conv_end = tf.keras.layers.Conv2D(30, 1, 1, padding='same', activation='sigmoid',
                                             kernel_regularizer=REGULARIZER)

    def set_config(self):
        # add regularizer
        self.resnet.trainable = True
        for layer in self.resnet.layers:
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', REGULARIZER)

        resnet_json = self.resnet.to_json()
        self.resnet = tf.keras.models.model_from_json(resnet_json)
        self.resnet.load_weights('/root/.keras/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                 by_name=True)


    def call(self, input_tensor, training=False):
        x = self.resnet(input_tensor, training=training)
        x = self.RB_0(x, training=training)
        x = self.conv_end(x)
        return x
