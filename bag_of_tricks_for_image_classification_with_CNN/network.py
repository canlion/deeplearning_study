import tensorflow as tf


REGULARIZER = tf.keras.regularizers.l2(1e-4)


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters, bn_gamma='ones', bn_beta='zeros', feature_halve=False, match_dims=False):
        """ResNet residual block."""
        super(ResidualBlock, self).__init__()

        self.match_dims = feature_halve or match_dims

        stride = 2 if feature_halve else 1
        self.conv_0 = tf.keras.layers.Conv2D(filters[0], 1, stride, 'same',
                                             kernel_regularizer=REGULARIZER,
                                             bias_regularizer=REGULARIZER)
        self.bn_0 = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                       gamma_initializer=bn_gamma)

        self.conv_1 = tf.keras.layers.Conv2D(filters[1], 3, 1, 'same',
                                             kernel_regularizer=REGULARIZER,
                                             bias_regularizer=REGULARIZER)
        self.bn_1 = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                       gamma_initializer=bn_gamma)

        self.conv_2 = tf.keras.layers.Conv2D(filters[2], 1, 1, 'same',
                                             kernel_regularizer=REGULARIZER,
                                             bias_regularizer=REGULARIZER)
        self.bn_2 = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                       gamma_initializer=bn_gamma)

        if self.match_dims:
            self.conv_sc = tf.keras.layers.Conv2D(filters[2], 1, stride, 'same',
                                                  kernel_regularizer=REGULARIZER,
                                                  bias_regularizer=REGULARIZER)
            self.bn_sc = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                            gamma_initializer=bn_gamma)

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


class Resnet50(tf.keras.Model):
    def __init__(self, bn_gamma='ones', bn_beta='zeros', category=1000):
        super(Resnet50, self).__init__()

        # stem
        self.conv_stem = tf.keras.layers.Conv2D(32, 7, 2, 'same',
                                                kernel_regularizer=REGULARIZER,
                                                bias_regularizer=REGULARIZER)
        self.bn_stem = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                          gamma_initializer=bn_gamma)
        self.pool_stem = tf.keras.layers.MaxPool2D(3, strides=2, padding='same')

        # stage_0
        self.RB_0_0 = ResidualBlock([32, 32, 128], match_dims=True)
        self.RB_0_1 = ResidualBlock([32, 32, 128])
        self.RB_0_2 = ResidualBlock([32, 32, 128])

        # stage_1
        self.RB_1_0 = ResidualBlock([64, 64, 256], feature_halve=True)
        self.RB_1_1 = ResidualBlock([64, 64, 256])
        self.RB_1_2 = ResidualBlock([64, 64, 256])
        self.RB_1_3 = ResidualBlock([64, 64, 256])

        # stage_2
        self.RB_2_0 = ResidualBlock([128, 128, 512], feature_halve=True)
        self.RB_2_1 = ResidualBlock([128, 128, 512])
        self.RB_2_2 = ResidualBlock([128, 128, 512])
        self.RB_2_3 = ResidualBlock([128, 128, 512])
        self.RB_2_4 = ResidualBlock([128, 128, 512])
        self.RB_2_5 = ResidualBlock([128, 128, 512])

        # stage_3
        self.RB_3_0 = ResidualBlock([256, 256, 1024], feature_halve=True)
        self.RB_3_1 = ResidualBlock([256, 256, 1024])
        self.RB_3_2 = ResidualBlock([256, 256, 1024])

        # classifier
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(category,
                                           kernel_regularizer=REGULARIZER,
                                           bias_regularizer=REGULARIZER)

    def call(self, input_tensor, training=False):
        # stem
        x = self.conv_stem(input_tensor)
        x = self.bn_stem(x, training=training)
        x = tf.keras.activations.relu(x)
        x = self.pool_stem(x)

        # stage_0
        x = self.RB_0_0(x, training=training)
        x = self.RB_0_1(x, training=training)
        x = self.RB_0_2(x, training=training)

        # stage_1
        x = self.RB_1_0(x, training=training)
        x = self.RB_1_1(x, training=training)
        x = self.RB_1_2(x, training=training)
        x = self.RB_1_3(x, training=training)

        # stage_2
        x = self.RB_2_0(x, training=training)
        x = self.RB_2_1(x, training=training)
        x = self.RB_2_2(x, training=training)
        x = self.RB_2_3(x, training=training)
        x = self.RB_2_4(x, training=training)
        x = self.RB_2_5(x, training=training)

        # stage_3
        x = self.RB_3_0(x, training=training)
        x = self.RB_3_1(x, training=training)
        x = self.RB_3_2(x, training=training)

        # classifier
        x = self.GAP(x)
        x = self.dense(x)
        x = tf.keras.activations.softmax(x)

        return x
