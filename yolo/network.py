import tensorflow as tf
import tensorflow.keras as K


REGULARIZER = tf.keras.regularizers.l2(5e-4)


def conv2d_BN_LReLU(filters, kernel, stride=1, alpha=.1):
    conv = K.layers.Conv2D(filters, kernel, stride, 'same',
                           kernel_regularizer=REGULARIZER,
                           kernel_initializer=K.initializers.he_normal())
    BN = K.layers.BatchNormalization()
    LReLU = K.layers.LeakyReLU(alpha=alpha)
    return conv, BN, LReLU


class YoloV1(K.Model):
    def __init__(self):
        super(YoloV1, self).__init__()

        self.seq_0 = K.Sequential([
            *conv2d_BN_LReLU(64, 7, 2),
            K.layers.MaxPool2D(padding='same'),
        ])

        self.seq_1 = K.Sequential([
            *conv2d_BN_LReLU(192, 3, 1),
            K.layers.MaxPool2D(padding='same'),
        ])

        self.seq_2 = K.Sequential([
            *conv2d_BN_LReLU(128, 1, 1),
            *conv2d_BN_LReLU(256, 3, 1),
            *conv2d_BN_LReLU(256, 1, 1),
            *conv2d_BN_LReLU(512, 3, 1),
            K.layers.MaxPool2D(padding='same'),
        ])

        self.seq_3 = K.Sequential([
            *conv2d_BN_LReLU(256, 1, 1),
            *conv2d_BN_LReLU(512, 3, 1),
            *conv2d_BN_LReLU(256, 1, 1),
            *conv2d_BN_LReLU(512, 3, 1),
            *conv2d_BN_LReLU(256, 1, 1),
            *conv2d_BN_LReLU(512, 3, 1),
            *conv2d_BN_LReLU(256, 1, 1),
            *conv2d_BN_LReLU(512, 3, 1),
            *conv2d_BN_LReLU(512, 1, 1),
            *conv2d_BN_LReLU(1024, 3, 1),
            K.layers.MaxPool2D(padding='same'),
        ])

        self.seq_4 = K.Sequential([
            *conv2d_BN_LReLU(512, 1, 1),
            *conv2d_BN_LReLU(1024, 3, 1),
            *conv2d_BN_LReLU(512, 1, 1),
            *conv2d_BN_LReLU(1024, 3, 2),
        ])

        self.seq_5 = K.Sequential([
            *conv2d_BN_LReLU(1024, 3, 1),
            *conv2d_BN_LReLU(1024, 3, 1),
            K.layers.Flatten(),
            K.layers.Dense(4096, kernel_regularizer=REGULARIZER),
            K.layers.LeakyReLU(alpha=.1),
            K.layers.Dropout(.5),
            K.layers.Dense(7*7*30, kernel_regularizer=REGULARIZER, activation='linear'),
        ])

    def call(self, input_tensor, training=False):
        x = self.seq_0(input_tensor, training=training)
        x = self.seq_1(x, training=training)
        x = self.seq_2(x, training=training)
        x = self.seq_3(x, training=training)
        x = self.seq_4(x, training=training)
        x = self.seq_5(x, training=training)
        x = tf.reshape(x, (-1, 7, 7, 30))

        return x


class YoloV1MobilenetV2(K.Model):
    def __init__(self):
        super(YoloV1MobilenetV2, self).__init__()

        self.net = K.applications.mobilenet_v2.MobileNetV2(input_shape=(448, 448, 3), include_top=False,
                                                           weights='imagenet')
        self.set_net()

        self.seq = K.Sequential([
            *conv2d_BN_LReLU(1024, 3, 1),
            *conv2d_BN_LReLU(1024, 3, 2),
            *conv2d_BN_LReLU(512, 3, 1),
            *conv2d_BN_LReLU(256, 3, 1),
            *conv2d_BN_LReLU(30, 1, 1),
            # K.layers.Flatten(),
            # K.layers.Dense(4096, kernel_regularizer=REGULARIZER),
            # K.layers.LeakyReLU(alpha=.1),
            # K.layers.Dropout(.5),
            # K.layers.Dense(7*7*30, kernel_regularizer=REGULARIZER),
        ])

    def set_net(self):
        # add regularizer
        self.net.trainable = True
        for layer in self.net.layers:
            if hasattr(layer, 'kernel_regularizer'):
                setattr(layer, 'kernel_regularizer', REGULARIZER)

        net_json = self.net.to_json()
        self.net = tf.keras.models.model_from_json(net_json)
        self.net.load_weights('/root/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
                              by_name=True)

    def call(self, input_tensor, training=False):
        x = self.net(input_tensor, training=training)
        x = self.seq(x, training=training)
        x = tf.reshape(x, (-1, 7, 7, 30))

        return x
