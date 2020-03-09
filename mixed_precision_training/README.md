# MIXED PRECISION TRAINING

paper : https://arxiv.org/pdf/1710.03740.pdf  
nvidia - Training with Mixed Precision : https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html

## 0. 논문읽기  
https://ai-study.tistory.com/37  

## 1. mixed precision training
 일반적으로 신경망 규모가 클수록 성능이 좋아지지만 당연히 메모리 필요량과 연산시간이 증가한다. 이러한 필요 자원의 증가를 돌거 위해
  데이터 타입을 단정밀도에서 반정밀도로 낮춰서 네트워크를 학습시키는 방법을 제시한다.
### 0. PF32 & FP16
