# ImageNet Classification with Deep Convolutional Neural Networks = AlexNet

## Abstract

+ non-saturating neuron들과 convolution operation에 매우 효율적인 GPU실행을 사용.
+ overfitting을 줄이기 위해 최근에 개발된 정규화 방법인 "dropout" 사용.
+ ILSVRC-2012 대회에 나와 15.3%의 에러율을 달성 (2등은 26.2%)

## Introduction

**즉, CNN을 통해 대규모 학습을 진행하고자 함**
+ 더 많은 training set 필요
    - 왜? 실제 상황에 있는 object는 수많은 변동성을 보여줌
    - 최근에야 큰 데이터 셋이 가능해짐
      - LabelMe : 다수의 fully-segmented image
      - ImageNet : 22000개의 카테고리안에 있는 1500만개의 라벨링된 높은 해상도 이미지
+ 많은 training set을 학습시키기 위해서, 학습을 많이 할 수 있는 model이 필요
+ 우리가 가지지 않은 모든 데이터를 보충할만한 기존의 지식이 많아야 함
  - 왜? object를 인식하는 일의 복잡성이 큼 -> 데이터 집합이 큼에도 불구하고 문제 지정 어려움
+ CNNs는 더 적은 connection과 parameter를 가지고 train 하기 쉬움
  - 이론상의 최고 성과는 약간만 나빠진다고 가정했을 경우, 유사한 사이즈의 layer와 표준적인 feedforward neural network와 비교할 시
  - 왜? Convolutional neural network는 가지지 않은 모든 데이터를 보충할 수 있는 클래스의 모델 중 하나로 구성됨
  - capacity(수용성)은 depth와 bredth를 다양하게 함으로서 조절 가능함
  - 이미지의 본질에 대해서 강하고 더 정확한 가정을 함
    - 이미지의 본질 = 통계학의 정상 과정과 픽셀 의존성의 지역성 (stationarity of statistics and locality of pixel dependencies)  
+ 하지만 여전히 이미지에 대규모로 적용하기엔 엄청나게 비쌈 -> 그래도 가능
  - 다행히도 현재 GPU는 대규모 CNNs의 훈련을 할 수 있을 정도로 강력함
    - 현재 GPU는 2D convolution의 매우 최적화된 구현과 잘 맞음
  - 최근의 데이터 셋에는 그러한 모델을 심각한 오버피팅 없이 훈련 시킬 수 있는 라벨이 부착

**이 논문의 특징**
1. ImageNet의 subset에서 가장 큰 CNN중의 하나를 훈련한 것
  - ILSVRC-2010과 -2012에서 사용되었던 ImageNet 사용
  - 이러한 dataset에서 가장 좋은 결과를 얻었다고 알려진 ImageNet 사용
  - 2D convolution에 매우 최적화 된 GPU 구현을 썼다.
  - public하게 사용 가능한, CNN을 학습하는데 본질적인 다른 모든 operation들을 사용
2. Network가 포함한 것
  - 성능을 향상시키고 훈련 시간을 줄이는 특징들의 갯수 [Section 3]
  - overfitting을 줄이기 위해 사용한 효과적인 기술들 [Section 4]
  - five convolutional + three fully-connected layers
  - depth : 모델 parameter의 1%밖에 포함하지 않는 convolutional layer 지움 => 더 나은 성능 추구
  - network size 제한
    - GPU의 메모리 한계 : GTX 580 3GB GPUs 
    - 기다릴 수 있는 시간의 한계 : 5-6일
    - 더 좋은 GPU와 더 많은 시간, 더 많은 데이터가 있으면 향상됨

## The Dataset
[ImageNet_fig2]
