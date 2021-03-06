# MIXED PRECISION TRAINING

paper : https://arxiv.org/pdf/1710.03740.pdf  
nvidia - Training with Mixed Precision : https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html

## Tensorflow mixed precision training
https://www.tensorflow.org/guide/keras/mixed_precision
```python
@tf.function
def train_network_mp(x, y):
    with tf.GradientTape() as tape:
        outp = model(tf.cast(x, tf.float16), training=True)
        
        # softmax는 FP32로 연산
        loss = loss_obj(y, tf.keras.activations.softmax(tf.cast(outp, tf.float32)))
        loss_l2 = loss + tf.math.add_n(model.losses)
        
        # loss scaling
        scaled_loss = optimizer.get_scaled_loss(loss_l2)
    
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # loss unscaling
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

...

if __name__ == '__main__':
	
    ...
	
    # setting mixed precision
    if args.mode == 'mixed_precision':
        # policy 선언 이후 생성되는 variable은 FP32로 저장되고 FP16으로 연산에 이용됨.
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
        print('compute : {}, variables : {}'.format(policy.compute_dtype, policy.variable_dtype))
        # >> compute : float16, variables : float32
        
        # 가속을 위해 XLA 컴파일러 이용 - 공부 필요!
        tf.config.optimizer.set_jit(True)
        
        # 옵티마이저를 loss scale 기능을 지원하도록 wrapping
        # loss scale 'dynamic' : 자동으로 loss scale factor를 지정.
        #     					 오버플로우시 감소시키고, 특정 스텝동안 오버플로가 없다면 증가시킴.
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
```


## 0. 논문읽기  
https://ai-study.tistory.com/37  

## 1. mixed precision training
 일반적으로 신경망 규모가 클수록 성능이 좋아지지만 당연히 메모리 필요량과 연산시간이 증가한다. 이러한 필요 자원의 증가를 덜기 위해 데이터 타입을 단정밀도에서 반정밀도로 낮춰서 네트워크를 학습시키는 방법을 제시한다.
### 1. FP32 & FP16
#### FP32
![fp32](https://user-images.githubusercontent.com/28844164/76207639-801cf880-6241-11ea-8e11-2bc3e364b5b9.png)
#### FP16
![fp16](https://user-images.githubusercontent.com/28844164/76207658-88753380-6241-11ea-9548-a57efa6d27ca.png)

### 2. mixed precision training을 위한 트릭들
#### 1. FP32 master weights
 네트워크 학습시에 그래디언트의 일부는 FP16 타입으로 표현할 수 있는 최소값보다 작아 0이 되어버린다. 또한 FP16은 가수부를 10비트로 표현하므로 지수부를 일치시켰을때 가중치와 그래디언트의 비의 비율이 1/2048보다 작다면 그래디언트는 0으로 처리된다. 따라서 FP32 타입의 가중치를 생성하고 forward, backward시에는 FP16 타입으로, 가중치 업데이트시에는 FP32 타입으로 수행한다. 
 
 
#### 2. loss scaling
 네트워크 학습시에 그래디언트의 일부는 FP16 타입으로 표현할 수 있는 최소값보다 작으므로 그래디언트에 loss scale factor를 곱해 FP16 타입으로 표현할 수 있는 범위로 밀어넣는다면 0이 되지 않는다. loss에 loss scale factor를 곱한다면 체인룰에 의해 모든 그래디언트는 loss scale factor만큼 scaling된다. FP32 master weights를 업데이트 하기전에 FP32로 형변환, unscaling을 수행해야한다.
#### 3. arithmetic precision
 몇몇 네트워크는 FP32 타입 학습의 정확도를 mixed training을 통해 재현하려면 FP16 벡터의 내적곱을 FP32 타입으로 합산해야한다(이유는 모르겠다. overflow문제인가??). 볼타 세대부터는 tensor core에 의해 자동으로 수행된다고 한다.
 
 
### 2. Result
* imagenet2012
* resnet-50 (OOM문제로 bottleneck구조의 3개 conv layer 중 마지막 layer의 필터수를 절반으로 감소시킴)
* 60 epoch / SGD(momentum 0.9, nesterov True) / lr 0.1 (cosine decay, warmup 3epoch)

거의 절반의 시간동안 네트워크 정확도를 재현할 수 있었음.
![mpt_result](https://user-images.githubusercontent.com/28844164/76208654-6f6d8200-6243-11ea-816a-2901fbadc674.png)
