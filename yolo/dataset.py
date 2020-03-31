import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

    
class VOCDatasetYOLOv1:
    def __init__(self, d_dir, valid_year, img_size=448, S=7, B=2, bbox_minimum=.025, aug=False):
        """
        tensorflow-datasets의 VOC 데이터셋 다운로드 및 데이터셋 객체 생성

        Arguments:
            d_dir - str / voc 데이터셋 저장 경로
            valid_year - int / validation셋을 이용할 voc 버전
            img_size - int / 이미지 리사이징 사이즈
            S - int / 이미지 분할 수 (S x S 구역으로 분할)
            B - int / 구역마다 예측할 바운딩박스 수
            bbox_minimum - float / 바운딩박스와 이미지의 비의 최소 비율 (최소 비율 미만인 박스는 삭제)
            aug - bool / train dataset augmentation 여부
        """
        self.train_ds, self.validation_ds, self.test_ds = None, None, None
        self.validation_num, self.train_num, self.test_num = 0, 0, 0
        self.set_dataset(valid_year, d_dir)

        self.img_size = img_size
        self.S = S
        self.B = B
        self.bbox_minimum = bbox_minimum
        self.aug = aug

        self.cls_list = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def set_dataset(self, valid_year, d_dir):
        """
        train 데이터셋과 validation 데이터셋 load.
        valid_year에 해당하는 voc 버전의 validation 데이터셋을 제외한 데이터셋은 학습에 사용됨.
        """
        
        assert valid_year in [2012, 2007]
        
        ds_2012, info_2012 = tfds.load('voc/2012', data_dir=d_dir, with_info=True)
        ds_2007, info_2007 = tfds.load('voc/2007', data_dir=d_dir, with_info=True)

        ds_dict = {2012: ds_2012, 2007: ds_2007}
        ds_num_dict = {2012: info_2012.splits, 2007: info_2007.splits}
        
        # 지정한 버전의 test dataset, validation dataset을 YOLO test dataset, validation dataset으로 사용
        self.validation_ds = ds_dict[valid_year]['validation']
        self.validation_num = ds_num_dict[valid_year]['validation'].num_examples
        self.test_ds = ds_dict[valid_year]['test']
        self.test_num = ds_num_dict[valid_year]['test'].num_examples
        
        # 지정한 validation dataset을 제외한 모든 train, validation dataset을 train dataset으로 사용
        train_ds_list = []
        for year in ds_dict.keys():
            for split in ds_dict[year].keys():
                if (year == valid_year and split == 'validation') or split == 'test':
                    continue
                train_ds_list.append(ds_dict[year][split])
                self.train_num += ds_num_dict[year][split].num_examples

        self.train_ds = train_ds_list[0]
        for ds in train_ds_list[1:]:
            self.train_ds = self.train_ds.concatenate(ds)

    def get_grid(self, bboxes, labels, flip=0, zoom=0, shift_x=0, shift_y=0):
        """
        YOLO 데이터 label 생성

        Arguments:
            bboxes - float / 바운딩박스의 이미지에 대한 상대적인 top, left, bottom, right
            labels - 1D-array(int) / 오브젝트의 카테고리
            flip - bool / 이미지 플립 여부
            zoom - float / 이미지 확대 비율
            shift_x - float / 이미지 가로 shift 비율
            shift_y - float / 이미지 세로 shift 비율

        Returns:
            grid - 3D-array / S x S 구역마다의 바운딩박스와 오브젝트 카테고리
        """
        grid = np.zeros((self.S, self.S, 25))
        for bbox, label in zip(bboxes, labels):
            # random crop & shift한 경우 바운딩박스 위치 조정
            # bbox : t, l, b, r
            bbox -= [zoom + shift_y, zoom + shift_x, zoom + shift_y, zoom + shift_x]
            
            # 이미지 리사이징시에 바운딩박스도 같이 확장
            factor = 1. / (1. - 2. * zoom)
            bbox = np.clip(bbox * factor, 0. + 1e-4, 1. - 1e-4)
            
            # 이미지 플립시에 바운딩박스 좌우 경계 교환
            if flip:
                bbox[1], bbox[3] = 1-bbox[3], 1-bbox[1]
                
            w, h = bbox[3]-bbox[1], bbox[2]-bbox[0]

            # 바운딩박스 크기가 전체 이미지에 대해 설정한 비율보다 작다면 삭제함
            # if w < self.bbox_minimum or h < self.bbox_minimum:
            #     continue
            
            # 바운딩박스 중심이 속하는 구역의 인덱스와 구역내에서의 상대적인 바운딩박스 중심 좌표
            grid_x, cx = divmod((bbox[3]+bbox[1])/2, 1/self.S)
            grid_y, cy = divmod((bbox[2]+bbox[0])/2, 1/self.S)

            grid_x, grid_y = int(grid_x), int(grid_y)
            # tf.print(grid_x, grid_y, self.cls_list[label])
            
            # 차례로 바운딩박스 중심, 가로세로, 오브젝트존재여부, 클래스 one-hot
            grid[grid_y, grid_x, 0] = cx * self.S
            grid[grid_y, grid_x, 1] = cy * self.S
            grid[grid_y, grid_x, 2] = np.sqrt(w)
            grid[grid_y, grid_x, 3] = np.sqrt(h)
            grid[grid_y, grid_x, 4] = 1.
            grid[grid_y, grid_x, label-20] = 1.

        return grid

    def img_augmentation(self, img, flip, zoom, shift_x, shift_y):
        """
        image augmentation 수행

        Arguments:
            img - 3D-array / 이미지
            flip - bool / 이미지 플립 여부
            zoom - float / 이미지 확대 비율
            shift_x - float / 이미지 가로 shift 비율
            shift_y - float / 이미지 세로 shift 비율

        Returns:
            img_aug - 3D-array / augmented image
        """
        # jpeg quality
        img_uint8 = tf.cast(img, tf.uint8)
        img_aug_uint8 = tf.image.random_jpeg_quality(img_uint8, 70, 100)
        img_aug = tf.cast(img_aug_uint8, img.dtype)

        # hue, brightness, saturation
        adjust_factor = tf.random.uniform((1, 1, 3), .5, 1.5)
        img_aug = tf.image.rgb_to_hsv(img_aug) * adjust_factor
        img_aug = tf.image.hsv_to_rgb(img_aug)

        # gauss noise
        noise = tf.random.normal(tf.shape(img_aug), 0., 0.1**(1/2))
        img_aug = tf.clip_by_value(img_aug + noise, 0., 255.)

        # 이미지를 랜덤한 비율만큼 확대, shift
        box = [zoom + shift_y, zoom + shift_x, 1. - zoom + shift_y, 1. - zoom + shift_x]
        img_aug = tf.image.crop_and_resize(img_aug[None, ...], [box], [0], [self.img_size, self.img_size])[0]

        if flip:
            img_aug = tf.image.flip_left_right(img_aug)

        return tf.squeeze(img_aug)

    def make_example(self, img, bbox, label, aug=False):
        """
        데이터 example 생성 - 이미지 augmentation & label 생성

        Arguments:
            img - 3D-array / 이미지
            bbox - 2D-array / 바운딩박스 좌표 (t, b, l, r)
            label - 1D-array / 오브젝트 카테고리
            aug - bool / augmentation 여부

        Returns:
             img - 3D-array / augmented 이미지 or 원본 이미지
             grid - 3D-array / S x S 구역마다의 바운딩박스와 오브젝트 카테고리
        """
        img = tf.cast(img, tf.float32)
        if aug:
            # 플립, 무작위 crop & resizing
            flip = np.random.randint(2)  # flip 여부
            zoom = np.random.uniform(0., .1)  # 이미지 가로세로 잘라낼 비율
            shift_x, shift_y = np.random.uniform(-zoom, zoom, 2)  # crop 위치
            img = self.img_augmentation(img, flip, zoom, shift_x, shift_y)
            grid = self.get_grid(bbox.numpy(), label.numpy(), flip, zoom, shift_x, shift_y)
        else:
            grid = self.get_grid(bbox.numpy(), label.numpy())

        img = tf.image.resize(img, (self.img_size,) * 2)
                
        return img, grid

    def dataset_train_batch(self, batch_size):
        """
        학습용 dataset 반환

        Arguments:
            batch_size - int / batch size

        Returns:
            ds - tf.data.dataset / image, label iterator - training dataset
            train_num - int / num of train data
        """
        # shuffle, map, batch, prefetch
        ds = self.train_ds.shuffle(2500)
        ds = ds.map(lambda x: tf.py_function(self.make_example,
                                             [x['image'], x['objects']['bbox'], x['objects']['label'], self.aug],
                                             [tf.float32, tf.float32],
                                             ),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds, self.train_num

    def dataset_valid_batch(self, batch_size):
        """
        검증용 dataset 반환
        
        Arguments:
            batch_size - int / batch size

        Returns:
            ds - tf.data.dataset / image, label iterator - validation dataset
            validation_num - int / num of validation data
        """
        ds = self.validation_ds.map(lambda x: tf.py_function(self.make_example,
                                                             [x['image'], x['objects']['bbox'], x['objects']['label']],
                                                             [tf.float32, tf.float32],
                                                             ),
                                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds, self.validation_num

    def dataset_test_batch(self):
        ds = self.test_ds.map(lambda x: (x['image'], x['image/filename']))

        return ds
