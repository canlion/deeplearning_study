import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2


class ImagenetDataset:
    def __init__(self, d_dir, ds_name='imagenet2012', print_info=False, label_depth=1000):
        """
        imagenet 데이터셋 생성

        :param d_dir: str / 이미지넷 데이터셋 저장 위치
        """
        ds, info = tfds.load(name=ds_name, data_dir=d_dir, with_info=True)
        self.train_ds, self.valid_ds = ds['train'], ds['validation']

        self.eigval = tf.constant([55.46, 4.794, 1.148])
        self.eigvec = tf.constant([[-0.5675, 0.7192, 0.4009],
                                   [-0.5808, -0.0045, -0.8140],
                                   [-0.5836, -0.6948, 0.4203]])
        self.mean = tf.constant([123.68, 116.779, 103.939], shape=(1, 1, 1, 3))
        self.std = tf.constant([58.393, 57.12, 57.375], shape=(1, 1, 1, 3))

        self.train_num = info.splits['train'].num_examples
        self.valid_num = info.splits['validation'].num_examples

        self.depth = label_depth

        if print_info:
            print(info)

    def get_train_ds(self, batch_size):
        """train 데이터셋 생성"""
        ds = self.train_ds.shuffle(10000)
        ds = ds.map(lambda x: (self._image_crop_train(x['image']), x['label']),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
        ds = ds.map(lambda x, y: self._make_example(x, y, True),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def get_valid_ds(self, batch_size):
        """validation 데이터셋 생성"""
        ds = self.valid_ds.map(lambda x: [self._image_crop_valid(x['image']), x['label']],
                               num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
        ds = ds.map(lambda x, y: self._make_example(x, y, False),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    @tf.function
    def _make_example(self, img, label, is_train):
        """
        imagenet tfrecord로부터 example 생성.

        :param img: 4D-array / (batch_size, h, w, ch) - 이미지 batch
        :param label: 1D-array / 이미지 class label batch
        :param is_train: bool / 이미지 augmentation 수행 여부
        :return: [4D-array, 2D-array] / 이미지 & label batch
        """
        if is_train:
            img = self._image_agumentation(img)
        label = tf.one_hot(label, depth=self.depth)

        return img, label

    @staticmethod
    def _image_crop_train(img):
        """
        무작위로 이미지 crop.
        crop 범위의 가로세로비와 크기는 랜덤하게 샘플링.

        :param img: 3D-array / (h, w, c) - 이미지
        :return: 3D-array / crop된 이미지
        """
        shape = tf.cast(tf.shape(img), tf.float32)
        h, w = shape[0], shape[1]
        area = h * w

        h_crop, w_crop = h, w
        # crop 범위가 원본 이미지 크기보다 작을때까지 새로운 범위 계산.
        while tf.logical_and(tf.less(h, h_crop), tf.less(w, w_crop)):
            aspect_ratio = tf.random.uniform([], 3 / 4, 4 / 3)
            area_ratio = tf.random.uniform([], 1 / 12.5, 1.)
            area_target = tf.multiply(area, area_ratio)
            h_crop = tf.sqrt(area_target / aspect_ratio)
            w_crop = h_crop * aspect_ratio

        img_crop = tf.image.random_crop(img, size=[h_crop, w_crop, 3])
        img_resize = tf.image.resize(img_crop, [224, 224])

        return tf.squeeze(img_resize)

    def _image_crop_valid(self, img):
        """
        validation 이미지 crop
        이미지의 비율은 유지한채로 짧은 축을 256 pixel로 리사이징 후
        이미지 중앙에서 224 x 224 crop.

        :param img: 3D-array / (h, w, c) - 이미지
        :return: 3D-array / crop된 이미지
        """
        h, w = tf.shape(img)[0], tf.shape(img)[1]
        size_new = [int(h*256/w), 256] if h > w else [256, int(w*256/h)]
        img = tf.image.resize(img, size_new)
        img_crop = tf.image.resize_with_crop_or_pad(img, 224, 224)
        img_normalize = (img_crop - self.mean) / self.std

        return tf.squeeze(img_normalize)

    def _image_agumentation(self, img):
        """
        trian 이미지 augmentation
        flip, hue, saturation, brightness, adding pca noise.

        :param img: 4D-array / (batch_size, h, w, ch) - 이미지 batch
        :return: 4D-array / augmented image (same shape as img)
        """
        # flip, hue, brightness, saturation
        img_aug = tf.image.random_flip_left_right(img)
        img_aug = tf.image.random_hue(img_aug, .4)
        img_aug = tf.image.random_brightness(img_aug, .4)
        img_aug = tf.image.random_saturation(img_aug, 0.6, 1.4)

        # add pca noise
        pca_alpha = tf.random.normal([3], 0., .1)
        pca_alpha_eigval = tf.multiply(pca_alpha, self.eigval)
        pca_noise = tf.tensordot(self.eigvec, pca_alpha_eigval, axes=1)
        img_aug += tf.reshape(pca_noise, (1, 1, 1, 3))

        # normalization
        img_aug = tf.clip_by_value(img_aug, 0., 255.)
        img_normalize = (img_aug - self.mean) / self.std

        return img_normalize

    def print_example(self):
        train_ds = self.get_train_ds(5).take(2)
        valid_ds = self.get_valid_ds(5).take(2)
        for type_, ds in zip(['train', 'valid'], [train_ds, valid_ds]):
            for i, (imgs, labels) in enumerate(ds):
                for j, (img, label) in enumerate(zip(imgs, labels)):
                    cv2.imwrite('{}_{}_{}.jpg'.format(type_, i, j), np.array(img[..., ::-1], dtype=np.int32))


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)
    inds = ImagenetDataset('/tf_datasets', ds_name='imagenette/full-size', print_info=True)
    # inds.print_example()
