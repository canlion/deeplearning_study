import tensorflow as tf
from utils import get_iou


class YOLOv1Loss(tf.keras.losses.Loss):
    def __init__(self, S=7, B=2,
                 lambda_coord=5., lambda_noobj=.5,
                 name='YOLO_loss'):
        """
        YOLO v1 loss

        Arguments:
            S - int / 이미지 분할 구역 수 (이미지를 S x S 구역으로 분할)
            B - int / 구역마다 predictor 수
            lambda_coord - float / localization loss 가중치
            lambda_noobj - float / object loss 중 오브젝트가 없는 구역에 대한 가중치
            name - str / 이름
        """
        super(YOLOv1Loss, self).__init__(reduction=tf.keras.losses.Reduction.AUTO, name=name)
        
        self.S = S
        self.B = B
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
    
    def call(self, true, pred):
        """
        YOLO v1 loss
        
        Arguments:
            grid - tensor / ground truth / (batch_size, S, S, 25)
            outp - tensor / prediction / (batch_size, S, S, 5*B+20)
        
        Returns:
            loss - tensor / loss / (,)
        """
        true_bbox, pred_bbox = true[..., :5], pred[..., :self.B*5]

        true_bbox = tf.reshape(true_bbox, (-1, self.S, self.S, 1, 5))
        pred_bbox = tf.reshape(pred_bbox, (-1, self.S, self.S, self.B, 5))

        # IOU
        iou = get_iou(true_bbox, pred_bbox, S=self.S, B=self.B)
        iou_one_hot = tf.one_hot(tf.argmax(iou, axis=-1), depth=self.B, axis=-1)

        # responsibility
        obj_true = true_bbox[..., 4]
        obj_1 = iou_one_hot * obj_true  # n, S, S, B
        no_obj_1 = 1. - obj_true  # n, S, S

        # bbox center loss
        xy, xy_hat = true_bbox[..., :2], pred_bbox[..., :2]
        xy_loss = tf.reduce_sum(obj_1[..., None] * tf.square(xy-xy_hat), axis=[1, 2, 3, 4])

        # bbox size loss
        wh, wh_hat = true_bbox[..., 2:4], pred_bbox[..., 2:4]
        wh_loss = tf.reduce_sum(obj_1[..., None] * tf.square(tf.sqrt(wh)-tf.sqrt(wh_hat)), axis=[1, 2, 3, 4])

        # confidence loss
        conf_hat = pred_bbox[..., 4]
        obj_loss = tf.reduce_sum(obj_1 * tf.square(1.-conf_hat), axis=[1, 2, 3])
        no_obj_loss = tf.reduce_sum(no_obj_1 * tf.square(conf_hat), axis=[1, 2, 3])

        # classification loss
        c, c_hat = true[..., -20:], pred[..., -20:]
        c_loss = tf.reduce_sum(obj_true * tf.square(c-c_hat), axis=[1, 2, 3])

        loss = self.lambda_coord * (xy_loss + wh_loss) + obj_loss + self.lambda_noobj * no_obj_loss + c_loss

        return tf.reduce_mean(loss)
