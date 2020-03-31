import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import math


def draw_bbox(grid, img, S=7):
    img = np.array(img, dtype=np.int32)
    
    h, w = img.shape[:2]
    obj_grid = np.array(np.where(grid[:, :, 4]==1.)).T

    edge_h, edge_w = img.shape[:2]
    edge_h, edge_w = int(edge_h/S), int(edge_w/S)
    for i in range(1, S):
        cv2.line(img, (0, edge_h*i), (img.shape[1], edge_h*i), (255, 0, 0), 1)
        cv2.line(img, (edge_w*i, 0), (edge_w*i, img.shape[0]), (255, 0, 0), 1)

    for y, x in obj_grid:
        cx, cy, bw, bh = grid[y, x, :4]
        cx = int((cx + x) / S*w)
        cy = int((cy + y) / S*h)
        bw = int(bw**2 * w)
        bh = int(bh**2 * h)
        print((cx-bw//2, cy-bh//2), (cx+bw//2, cy+bh//2))
        
        img = cv2.rectangle(img,
                            (cx-bw//2, cy-bh//2), (cx+bw//2, cy+bh//2),
                            (0, 255, 255), 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()
    
    
def draw_predicted_bbox(pred, img):
    img = np.array(img, dtype=np.int32)
    pred = np.array(pred, dtype=np.float32)
    h, w = img.shape[:2]
    S = pred.shape[0]
    
    for y in range(S):
        for x in range(S):
            data = pred[y, x]
            if data[4] > data[9]:
                cx, cy, bw, bh = data[:4]
                C = data[4]
                color = (255, 0, 0)
            else:
                cx, cy, bw, bh = data[5:9]
                C = data[9]
                color = (0, 255, 0)

            bw, bh = bw**2, bh**2
            cx = int((cx+x)/S * w)
            cy = int((cy+y)/S * h)
            bw = int(bw*w)
            bh = int(bh*h)
            
            if C > .5:
                img = cv2.rectangle(img,
                                    (cx-bw//2, cy-bh//2), (cx+bw//2, cy+bh//2),
                                    color, int(5*C))
                
    plt.imshow(img)
    plt.show()


def nms(pred):
    # TODO : non maximum suppression 작성
    # pred (S, S, 5*B + 20)
    pass


def get_iou(box_1, box_2, S=7, B=2):
    box_1 = tf.tile(box_1, [1, 1, 1, B, 1])
    box_stack = tf.stack([box_1, box_2], axis=-1)  # n, s, s, 2, 5, 2

    xy = box_stack[..., :2, :]
    wh = tf.square(box_stack[..., 2:4, :])
    small_wh = tf.reduce_min(wh, axis=-1)

    inter_lt = tf.reduce_max(xy/S - wh/2, axis=-1)
    inter_rb = tf.reduce_min(xy/S + wh/2, axis=-1)
    inter_wh = tf.clip_by_value(inter_rb - inter_lt, 0., small_wh)  # n, s, s, 2, 2
    intersection = tf.reduce_prod(inter_wh, axis=-1)

    union = tf.reduce_sum(tf.reduce_prod(wh, axis=-2), axis=-1) - intersection + 1e-10

    return tf.cast(tf.clip_by_value(intersection / union, 0., 1.), tf.float32)


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


class LRWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, scheduler, step, init_lr=0):
        super(LRWarmup, self).__init__()
        assert step > 0, 'learning rate warm-up step must be larger than zero.'

        self.init_lr = init_lr
        self.scheduler = scheduler
        self.lr = self.scheduler(0)
        self.step = step

    @tf.function
    def __call__(self, steps):
        if tf.less_equal(steps, self.step):
            return self.init_lr + (self.lr-self.init_lr) * (steps / self.step)
        else:
            return self.scheduler(steps - self.step)


if __name__ == '__main__':
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([5, 10], [.1, .01, .001])
    wu_lr = LRWarmup(lr, 5)

    for step in range(20):
        print(wu_lr(step))
