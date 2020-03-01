import tensorflow as tf
import math


class WarmupExponential(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, init_lr, decay_steps, decay_rate, staircase=False, warmup_steps=0):
        """
        learning rate exponential decay에 learning rate warmup 적용.

        ```python
        if steps <= warmup_steps:
            return init_lr * (steps / warmup_steps)
        else:
            exponential = floor(steps / decay_steps) if staircase else steps / decay_steps
            return init_lr * decay_rate ** exponential
        ```

        :param init_lr: exponential decay의 초기 lr
        :param decay_steps: decay_rate 지수의 분모
        :param decay_rate: lr을 감소시키는 비율
        :param staircase: decay_rate의 지수를 floor할지 여부. 적용시 lr은 계단함수
        :param warmup_steps: lr을 0부터 init_lr까지 선형적으로 증가시킬 스텝 수
        """
        super(WarmupExponential, self).__init__(
            initial_learning_rate=init_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
        )
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps

    @tf.function
    def __call__(self, steps):
        if tf.math.less_equal(steps, self.warmup_steps):
            return self.init_lr * (steps / self.warmup_steps)
        else:
            return super(WarmupExponential, self).__call__(steps)


class WarmupCosineLRDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, total_batch, warmup_steps=0):
        """
        learning rate cosine decay & learning rate warmup

        :param init_lr: cosine decay의 초기 lr
        :param total_batch: cosine decay를 적용할 총 스텝 수
        :param warmup_steps: lr을 0부터 init_lr까지 선형적으로 증가시킬 스텝 수
        """
        super(WarmupCosineLRDecay, self).__init__()
        self.init_lr = init_lr
        self.T = total_batch - warmup_steps
        self.warmup_steps = warmup_steps

    @tf.function
    def __call__(self, steps):
        if tf.math.less_equal(steps, self.warmup_steps):
            return self.init_lr * (steps / self.warmup_steps)
        else:
            steps_after_warmup = steps - self.warmup_steps
            cur_lr = self.init_lr * (1 + tf.math.cos(steps_after_warmup * math.pi / self.T)) / 2
            return cur_lr


@tf.function
def label_smoothing(one_hot, epsilon=.1, K=10):
    """
    one-hot 벡터에 label smoothing 적용

    :param one_hot: one-hot 벡터
    :param epsilon: label smoothing 계수
    :param K: one-hot 벡터의 카테고리 수
    :return: label smoothed 벡터
    """
    label_smooth = one_hot * ((1 - epsilon) - (epsilon / (K-1)))
    label_smooth += epsilon / (K-1)
    return label_smooth


@tf.function
def mix_up(img_batch, label_batch, alpha=.2):
    """
    이미지와 레이블에 mixup augmentation 적용
    Beta(alpha, alpha) 분포에서 샘플링한 값을 가중치로 하여 두 이미지, 레이블을 선형 보간.

    :param img_batch: 이미지 배치
    :param label_batch: 레이블 배치
    :param alpha: Beta 분포 계수
    :return: mixup된 이미지, 레이블 배치쌍
    """
    batch_size = tf.shape(img_batch)[0]
    lambda_ = tf.compat.v1.distributions.Beta(alpha, alpha).sample([batch_size])

    lambda_img = tf.reshape(lambda_, (batch_size, 1, 1, 1))
    lambda_label = tf.reshape(lambda_, (batch_size, 1))

    perm = tf.random.shuffle(tf.range(batch_size))
    img_batch_shuffle = tf.gather(img_batch, perm, axis=0)
    label_batch_shuffle = tf.gather(label_batch, perm, axis=0)

    img_batch_mixup = lambda_img * img_batch + (1-lambda_img) * img_batch_shuffle
    label_batch_mixup = lambda_label * label_batch + (1-lambda_label) * label_batch_shuffle

    return img_batch_mixup, label_batch_mixup


if __name__ == '__main__':
    # we = WarmupExponential(.1, warmup_steps=5, decay_steps=10, decay_rate=.1, staircase=True)
    # for step in range(0, 60, 5):
    #     print('{} steps - lr : {}'.format(step, we(step)))

    print(label_smoothing(tf.constant([[0, 0, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=tf.float32), K=5))
