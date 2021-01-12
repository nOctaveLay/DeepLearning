# Semantic Image Segmentation With Deep Convolutional Nets and Fully Connected CRFs

## Index

1. [Abstract](##Abstract)
2. [Related Work](##Related%20Work)
3. [Methods](##Methods)
4. [Experimental Results](##Experimental%20Results)
5. [Conclusion](##Conclusion)

## Abstract

- 2016년 7월 7일 논문
- [출처](https://arxiv.org/pdf/1412.7062.pdf)
- DCNN은 image classification과 object detection에 있어 high level vision task에서 좋은 성과를 내고 있다.
  - 이는 DCNN에서 method를 가지고 오게 되는 계기가 되었다.
- pixel-level classification의 task(다른 말로 semantic image segmentation)를 설명하기 위해서 probabilistic graphical model을 가져왔다.
  - 이것은 high level task에서 DCNN을 좋게 만드는 invariance 속성 때문이다.
- 마지막 DCNN layer에 대한 response들을 fully connected Conditional Random Field (CRF)와 결합시킴으로서, deep network의 poor localization property를 극복한다.
- 질적으로, 우리의 "DeepLab" 시스템은 segment boundary들을 과거의 모델을 뛰어넘은 정확성으로 localize 할 수 있다.
  - PASCAL VOC-2012 semantic image segmentation task에서 우리의 방법을 설정했을 때, test set에서 71.6%의 IOU accuracy을 잡았다.
- 어떻게 이런 결과가 효율적으로 얻어질 수 있는지를 보여준다.
  - 신중하게 network 재 목적화
  - wavelet community에서 나온 "hole" algorithm의 새로운 적용
  - 이 둘은 neural net의 dense computation이 1초마다 8프레임의 응답을 내도록 한다.

## Introduction

- DCNN은 지난 2년동안 high-level 문제들에 대해 computer vision의 성능을 높이도록 push받아왔다.
  - 이 문제들은 image classification, object detection, fine-grained categorization 이 셋 중에 하나다.
  - 이러한 work들의 일반적인 테마는 end-to-end방식으로 학습된 DCNN이 놀랍게도 조심스럽게 engineer된 표현들에 의존하는 system보다 더 나은 결과를 전달한다는 것이었다.
  - 이런 성공은 local image transformation에서 DCNN의 built-in invariance에 기여되었고, 이것은 data의 hierarchical abstraction(계층 추상화)를 배우도록 하는 그들의 능력을 뒷받침했다.
  - Invariance -> high level vision task에서 명백히 요구되는 것, 그러나 low-level task에 방해가 됨(예를 들어 pose estimation, semantic segmentation)
    - low-level task는 spatial detail보다 정확한 localization이 필요하기 때문.
- DCNN을 image labeling task에 적용하는 데에 있어 2개의 기술적인 장애물 : signal down-sampling, spatial 'insensitivity' (invariance).
  - signal down-sampling
    - standard DCNN의 모든 layer에서 수행되는 max-pooling과 downsampling('striding')의 반복적인 조합으로 인해 일어남.
    - 이를 해결하기 위해, 'atrous' algorithm을 적용.
    - 효율적인 dense computation을 이전의 해결책보다 간단하게 만들어줌
  - spatial 'insensitivity' (invariance)
    - classifier에서 object 중심의 decision들을 얻는 것이 spatial transformation에 있어 invariance를 요구하기 때문에 일어남.
    - 이는 본질적으로 DCNN 모델의 spatial accuracy를 제한시킴.
    - Fully-connected CRF를 적용시킴으로서 fine detail을 잡는 모델의 능력을 향상시킴.
- CRF
  - semantic segmentation에서 multi-way classifier에 의해 계산된 class score를 edge와 pixel 또는 superpixel들의 local interaction들에 의해 얻어진 low-level information과 결합시키는 데 널리 사용되어 왔음.
  - 증가된 세밀함(sophistication)에 대한 작업들이 계층적 의존(hierarchical dependency)와 segment들의 높은 순서 의존도(high-order depencdency of segments)을 제안해왔다.
  - 하지만 효율적인 계산과 long range dependency를 가지면서 fine edge detail들을 잡는 능력때문에 fully connected pairwise CRF를 사용하기로 함.
  - 이 모델은 Krahenbuhl & Koltun의 논문에서 pixel-level classifier에 기초해 성능을 크게 향상시키기 위해서 제시되었다.
  - 그리고 우리의 DeepLab에서 DCNN을 기초로 한 pixel-level classifier와 결합되었을 때 최신의 결과를 이끈다라는 것을 설명할 것이다.
- DeepLab system의 3가지 장점 : speed, accuracy, simplicity
  - speed : atrous algorithm 때문에, fully-connected CRF를 위한 Mean Field Inference가 0.5초를 필요로 하지만, 우리의 dense DCNN은 8fps로 작동한다.
  - accuracy : Mostajabi et al의 2번째로 뛰어난 접근을 뛰어넘으면서 7.2%의 격차를 가지고 PASCAL semantic segmentation challenge에서 최신의 결과를 얻었다.
  - simplicity : 2개의 합리적으로 잘 설계된 모듈인 DCNN과 CRF의 계단(cascade)으로 구성되어있다.

## Related Work
