import tensorflow as tf


REGULARIZER = tf.keras.regularizers.l2(1e-4)


#########
# block #
#########
class ResidualBlockBNeck(tf.keras.Model):
    def __init__(self, filters, bn_gamma='ones', bn_beta='zeros',
                 weight_regularizer=REGULARIZER, bias_regularizer=REGULARIZER,
                 feature_halve=False, match_dims=False):
        """ResNet residual block."""
        super(ResidualBlockBNeck, self).__init__()

        self.match_dims = feature_halve or match_dims

        stride = 2 if feature_halve else 1
        self.conv_0 = tf.keras.layers.Conv2D(filters[0], 1, stride, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer)
        self.bn_0 = tf.keras.layers.BatchNormalization()

        self.conv_1 = tf.keras.layers.Conv2D(filters[1], 3, 1, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer)
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(filters[2], 1, 1, 'same',
                                             kernel_regularizer=weight_regularizer,
                                             bias_regularizer=bias_regularizer)
        self.bn_2 = tf.keras.layers.BatchNormalization(beta_initializer=bn_beta,
                                                       gamma_initializer=bn_gamma)

        if self.match_dims:
            self.conv_sc = tf.keras.layers.Conv2D(filters[2], 1, stride, 'same',
                                                  kernel_regularizer=weight_regularizer,
                                                  bias_regularizer=bias_regularizer)
            self.bn_sc = tf.keras.layers.BatchNormalization()

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


class ResidualBlock_BN_D(tf.keras.Model):
    def __init__(self, filters, bn_gamma='ones', bn_beta='zeros',
                 weight_regularizer=REGULARIZER, bias_regularizer=REGULARIZER,
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
        x = tf.keras.activations.relu(x)

        x = self.conv_1(x)
        x = self.bn_1(x, training=training)
        x = tf.keras.activations.relu(x)

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
        x = tf.keras.activations.relu(x)

        return x


###########
# network #
###########
class Resnet_Cifar(tf.keras.Model):
    def __init__(self, bn_gamma='ones', bn_beta='zeros', category=10,
                 weight_regularizer=REGULARIZER, bias_regularizer=REGULARIZER):
        super(Resnet_Cifar, self).__init__()

        # stem
        self.conv_stem = tf.keras.layers.Conv2D(32, 3, 1, 'same',
                                                kernel_regularizer=REGULARIZER,
                                                bias_regularizer=REGULARIZER)
        self.bn_stem = tf.keras.layers.BatchNormalization()

        # stage_0
        self.RB_0_0 = ResidualBlockBNeck([32, 32, 128], match_dims=True,
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_1 = ResidualBlockBNeck([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_2 = ResidualBlockBNeck([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_3 = ResidualBlockBNeck([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_4 = ResidualBlockBNeck([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)

        # stage_1
        self.RB_1_0 = ResidualBlockBNeck([64, 64, 256], feature_halve=True,
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_1 = ResidualBlockBNeck([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_2 = ResidualBlockBNeck([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_3 = ResidualBlockBNeck([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_4 = ResidualBlockBNeck([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)

        # stage_2
        self.RB_2_0 = ResidualBlockBNeck([128, 128, 512], feature_halve=True,
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_1 = ResidualBlockBNeck([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_2 = ResidualBlockBNeck([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_3 = ResidualBlockBNeck([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_4 = ResidualBlockBNeck([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)

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

        # stage_0
        x = self.RB_0_0(x, training=training)
        x = self.RB_0_1(x, training=training)
        x = self.RB_0_2(x, training=training)
        x = self.RB_0_3(x, training=training)
        x = self.RB_0_4(x, training=training)

        # stage_1
        x = self.RB_1_0(x, training=training)
        x = self.RB_1_1(x, training=training)
        x = self.RB_1_2(x, training=training)
        x = self.RB_1_3(x, training=training)
        x = self.RB_1_4(x, training=training)

        # stage_2
        x = self.RB_2_0(x, training=training)
        x = self.RB_2_1(x, training=training)
        x = self.RB_2_2(x, training=training)
        x = self.RB_2_3(x, training=training)
        x = self.RB_2_4(x, training=training)

        # classifier
        x = self.GAP(x)
        x = self.dense(x)
        x = tf.keras.activations.softmax(x)

        return x


class Resnet_Cifar_Tweak(tf.keras.Model):
    def __init__(self, bn_gamma='ones', bn_beta='zeros', category=10,
                 weight_regularizer=REGULARIZER, bias_regularizer=REGULARIZER):
        """ResNet-D"""
        super(Resnet_Cifar_Tweak, self).__init__()

        # stem
        self.conv_stem = tf.keras.layers.Conv2D(32, 3, 1, 'same',
                                                kernel_regularizer=REGULARIZER,
                                                bias_regularizer=REGULARIZER)
        self.bn_stem = tf.keras.layers.BatchNormalization()

        # stage_0
        self.RB_0_0 = ResidualBlock_BN_D([32, 32, 128], match_dims=True,
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_1 = ResidualBlock_BN_D([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_2 = ResidualBlock_BN_D([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_3 = ResidualBlock_BN_D([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_0_4 = ResidualBlock_BN_D([32, 32, 128],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)

        # stage_1
        self.RB_1_0 = ResidualBlock_BN_D([64, 64, 256], feature_halve=True,
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_1 = ResidualBlock_BN_D([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_2 = ResidualBlock_BN_D([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_3 = ResidualBlock_BN_D([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_1_4 = ResidualBlock_BN_D([64, 64, 256],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)

        # stage_2
        self.RB_2_0 = ResidualBlock_BN_D([128, 128, 512], feature_halve=True,
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_1 = ResidualBlock_BN_D([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_2 = ResidualBlock_BN_D([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_3 = ResidualBlock_BN_D([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)
        self.RB_2_4 = ResidualBlock_BN_D([128, 128, 512],
                                         bn_gamma=bn_gamma, bn_beta=bn_beta,
                                         weight_regularizer=weight_regularizer,
                                         bias_regularizer=bias_regularizer)

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

        # stage_0
        x = self.RB_0_0(x, training=training)
        x = self.RB_0_1(x, training=training)
        x = self.RB_0_2(x, training=training)
        x = self.RB_0_3(x, training=training)
        x = self.RB_0_4(x, training=training)

        # stage_1
        x = self.RB_1_0(x, training=training)
        x = self.RB_1_1(x, training=training)
        x = self.RB_1_2(x, training=training)
        x = self.RB_1_3(x, training=training)
        x = self.RB_1_4(x, training=training)

        # stage_2
        x = self.RB_2_0(x, training=training)
        x = self.RB_2_1(x, training=training)
        x = self.RB_2_2(x, training=training)
        x = self.RB_2_3(x, training=training)
        x = self.RB_2_4(x, training=training)

        # classifier
        x = self.GAP(x)
        x = self.dense(x)
        x = tf.keras.activations.softmax(x)

        return x
