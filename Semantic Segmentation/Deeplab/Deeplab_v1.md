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


