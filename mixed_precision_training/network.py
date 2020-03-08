import tensorflow as tf


REGULARIZER = tf.keras.regularizers.l2(1e-4)


#########
# block #
#########
class ResidualBlockBNeck(tf.keras.Model):
    def __init__(self, filters, bn_gamma='ones', bn_beta='zeros',
                 weight_regularizer=REGULARIZER, bias_regularizer=None,
                 beta_regularizer=None, gamma_regularizer=REGULARIZER,
                 feature_halve=False, match_dims=False, **kwargs):
        """ResNet residual block."""
        super(ResidualBlockBNeck, self).__init__()

        self.match_dims = feature_halve or match_dims

        stride = 2 if feature_halve else 1
        self.conv_0 = tf.keras.layers.Conv2D(filters[0], 1, stride, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer, **kwargs)
        self.bn_0 = tf.keras.layers.BatchNormalization(beta_regularizer=beta_regularizer,
                                                       gamma_regularizer=gamma_regularizer)

        self.conv_1 = tf.keras.layers.Conv2D(filters[1], 3, 1, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer, **kwargs)
        self.bn_1 = tf.keras.layers.BatchNormalization(beta_regularizer=beta_regularizer,
                                                       gamma_regularizer=gamma_regularizer)

        self.conv_2 = tf.keras.layers.Conv2D(filters[2], 1, 1, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer, **kwargs)
        self.bn_2 = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                       gamma_initializer=bn_gamma,
                                                       beta_regularizer=beta_regularizer,
                                                       gamma_regularizer=gamma_regularizer)

        if self.match_dims:
            self.conv_sc = tf.keras.layers.Conv2D(filters[2], 1, stride, 'same',
                                                  kernel_regularizer=weight_regularizer,
                                                  bias_regularizer=bias_regularizer,
                                                  **kwargs)
            self.bn_sc = tf.keras.layers.BatchNormalization(beta_regularizer=beta_regularizer,
                                                            gamma_regularizer=gamma_regularizer)

    def call(self, input_tensor, training=False):
        x = self.conv_0(input_tensor)
        x = self.bn_0(x, training=training)
        x = tf.keras.activations.relu(x)

        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = tf.keras.activations.relu(x)

        x = self.conv_2(x)
        x = self.bn_2(x, training=training)

        if self.match_dims:
            shortcut = self.conv_sc(input_tensor)
            shortcut = self.bn_sc(shortcut, training=training)
        else:
            shortcut = input_tensor

        x = x + shortcut
        x = tf.keras.activations.relu(x)

        return x

###########
# network #
###########

class Resnet50Half(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Resnet50Half, self).__init__()
        # stem
        self.conv_stem = tf.keras.layers.Conv2D(64, 7, 2, 'same', kernel_regularizer=REGULARIZER, **kwargs)
        self.bn_stem = tf.keras.layers.BatchNormalization(gamma_regularizer=REGULARIZER)
        self.pool_stem = tf.keras.layers.MaxPool2D((2, 2), 2, 'same')

        # stage 0
        self.RB_0_0 = ResidualBlockBNeck([64, 64, 128], match_dims=True, **kwargs)
        self.RB_0_1 = ResidualBlockBNeck([64, 64, 128], **kwargs)
        self.RB_0_2 = ResidualBlockBNeck([64, 64, 128], **kwargs)

        # stage 1
        self.RB_1_0 = ResidualBlockBNeck([128, 128, 256], feature_halve=True, **kwargs)
        self.RB_1_1 = ResidualBlockBNeck([128, 128, 256], **kwargs)
        self.RB_1_2 = ResidualBlockBNeck([128, 128, 256], **kwargs)
        self.RB_1_3 = ResidualBlockBNeck([128, 128, 256], **kwargs)

        # stage 2
        self.RB_2_0 = ResidualBlockBNeck([256, 256, 512], feature_halve=True, **kwargs)
        self.RB_2_1 = ResidualBlockBNeck([256, 256, 512], **kwargs)
        self.RB_2_2 = ResidualBlockBNeck([256, 256, 512], **kwargs)
        self.RB_2_3 = ResidualBlockBNeck([256, 256, 512], **kwargs)
        self.RB_2_4 = ResidualBlockBNeck([256, 256, 512], **kwargs)
        self.RB_2_5 = ResidualBlockBNeck([256, 256, 512], **kwargs)

        # stage 3
        self.RB_3_0 = ResidualBlockBNeck([512, 512, 1024], feature_halve=True, **kwargs)
        self.RB_3_1 = ResidualBlockBNeck([512, 512, 1024], **kwargs)
        self.RB_3_2 = ResidualBlockBNeck([512, 512, 1024], **kwargs)

        # classifier
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(1000, kernel_regularizer=REGULARIZER, **kwargs)

    def call(self, input_tensor, training=False):
        # stem
        x = self.conv_stem(input_tensor)
        x = self.bn_stem(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self.pool_stem(x)

        # stage 0
        x = self.RB_0_0(x, training=training)
        x = self.RB_0_1(x, training=training)
        x = self.RB_0_2(x, training=training)

        # stage 1
        x = self.RB_1_0(x, training=training)
        x = self.RB_1_1(x, training=training)
        x = self.RB_1_2(x, training=training)
        x = self.RB_1_3(x, training=training)

        # stage 2
        x = self.RB_2_0(x, training=training)
        x = self.RB_2_1(x, training=training)
        x = self.RB_2_2(x, training=training)
        x = self.RB_2_3(x, training=training)
        x = self.RB_2_4(x, training=training)
        x = self.RB_2_5(x, training=training)

        # stage 3
        x = self.RB_3_0(x, training=training)
        x = self.RB_3_1(x, training=training)
        x = self.RB_3_2(x, training=training)

        # classifier
        x = self.GAP(x)
        x = self.dense(x)

        return x
