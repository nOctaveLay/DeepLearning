# Semantic Image Segmentation With Deep Convolutional Nets and Fully Connected CRFs

## Index

1. [Abstract](##Abstract)
2. [Related Work](##Related%20Work)
3. [Convolutional Neural Networks for Dense Image Labeling](##Convolutional%20Neural%20Networks%20for%20Dense%20Image%20Labeling)
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

## Convolutional Neural Networks for Dense Image Labeling

- 밑은 우리의 dense semantic image segmentation 시스템을 위해 public하게 이용 가능한 미리 학습시켜놓은 최신의 16 layer classification network(VGG-16)을 효율적, 효과적인 dense feature extractor로 어떻게 목적을 다시 설정했고 어떻게 finetune 시켰는지를 설명한다.

### Efficient Dense Sliding Window Feature Extraction with the Hole Algorithm

- 우리의 dense CNN feature extractor의 성공에서 Dense spatial score 평가는 중요하다.
  - 이를 실행하기 위해서, VGG-16의 fully-connected layer들을 convolution한 것으로 변환한다.
  - 그리고 이미지의 original resolution에서 convolutional 방법으로 network를 돌린다.
  - 하지만 이러한 방법이 드문 드문 계산된 detection scores를 산출하기 때문에 충분하지 않다. (32 pixel의 stride를 갖는다.)
- 우리의 target인 8픽셀 stride를 가진 score를 더 densely하게 계산하기 위해서, Giusti et al.이 이전에 적용시킨 방법의 변형을 개발했다.
  - Simonyan & Zisseman (2014)의 network에서 마지막 두 개의 max-pooling layer들 후에 이뤄지는 subsampling을 skip했다. 그리고 길이를 늘리기 위해 zero들을 집어넣은 방식을 택한 layer 안에서 convolutional filter를 수정했다.(마지막 3개의 convolutional layer에서는 2배, 첫 번째 fully connected layer는 4배)
  - filter들을 온전한 상태로 유지함으로서 이를 효율적으로 실행함
  - filter들을 각각 2개 또는 4개의 픽셀의 input stride를 각각 사용했을 때 적용된 feature map들을 드문드문 대신 sample했다. => atrous convolution
  - 일반적으로 적용되고, 어떤 근사치를 소개하는 일 없이 어떤 target subsampling rate에서도 dense CENN feature map을 효율적으로 계산할 수 있도록 만든다.
- 이미지 넷을 미리 학습시킨 VGG network의 모델 weight을 finetune 시킨다.
  - 이는 직접적으로 image classification task에 적용시키기 위해서이다.
  - 이는 Long et al.의 과정을 따른다. (2014)
  - 1000 개의 방법으로 되어 있는 VGG-16의 마지막 레이어의 Imagenet classifier를 21가지 방법으로 대체했다.
  - loss function은 (original image와 비교해서 8로 subsample된) CNN output map에서 각각의 spatial position을 위해 cross-entropy의 합으로 계산했다.
  - 모든 포지션과 라벨들은 전반적인 loss function 안에서 동등하게 weight를 매겼다.
  - 우리의 목표는 (8로 서브샘플링 된) ground truth label들이다.
  - 최적화는 standard SGD procedure를 사용해서 모든 network layer의 weight에 대해 objective function을 사용할 것이다.
- 테스트 동안, 우리는 original image resolution에 대해 class score map이 필요하다.
  - Figure 2와 Section 4.1에서 볼 수 있듯, class score map이 (log-probabilities를 따른다.) 꽤 smooth 하다. 이는 simple bilinear interpolation을 타협할 수 있는 비용으로 이미지의 해상도를 8배 증가시키기 위해 사용할 수 있다고 할 수 있다.
  - Long et al.의 방법은 hole algorithm을 사용하지 않았고, 그래서 매우 거친 score가 생산되었다. (32배로 subsample 되었음)
  - 이 사실은 학습된 upsampling layer를 사용하라고 강요했고, 이는 복잡성과 그들의 시스템의 훈련 시간을 증가시켰다.
- 잘 tuning 된 우리의 network는 PASCAL VOC 2012에서 10시간이나 걸렸고, 그들은 훈련하는 데 며칠의 시간이 걸렸다고 한다. (both timing on a modern GPU)

### Controlling the Receptive Field Size and Accelerating Dense Computation with Convolutional Nets

> Receptive Field의 크기를 조절하고, Convolutional NEt에서 깊은 계산을 증폭시키기

- 우리의 network를 dense score computation을 위해서 목적을 다시 설정하게 하는 또 다른 요소는 network의 receptive field size를 명백히 control하기 위해서이다.
- 대부분 최신의 DCNN에 기반을 둔 이미지 인식 방법은 Imagenet large-scale classification task에서 미리 훈련된 network에게 의존한다.
- 이러한 network들은 일반적으로 거대한 receptive field size를 가진다.
  - 우리는 VGG-16 net을 고려했다.
    - 이 넷의 receptive field size는 224 x 224이다.
      - (zero padding을 고려한다.)
    - (Convolution하게 계산했을 경우) 404 x 404 픽셀들이다.
  - network를 fully convolutional 한 것으로 바꾼 후
    - 첫번째 fully connected layer
      - 4096개의 filter와 커다란 7 x 7 spatial size를 가진다.
      - 우리의 dense score map computation에서 computational bottleneck이 걸린다.
- 첫번째 FC 레이어를 4x4 spatial size로 spatially subsampling (by simple decimation)함으로서 이 문제를 해결
  - receptive field를 128x128(with zero padding)또는 308x308(in convolutional mode)로 줄였다.
  - 첫 번째 FC layer들을 계산 하는 데 걸리는 시간 (computation time)을 2-3배 정도 줄였다.
  - Caffe로 구현된 것과 Titan GPU를 사용함으로써 VGG에서 나온 network가 매우 효율적인 결과를 보여줬다.
    - 306x306 input image가 주어졌을 때, network의 맨 위에 있는 부분에서 testing동안 약 8 frames/sec의 비율로 39x39 dense raw feature score들을 보여줬다.
    - training 동안 스피드는 3 frames/sec이다.
    - 우리는 성공적으로 channel들의 개수를 4096개에서 1024개로 줄이는 것을 실험소다.
    - 이는 성능이 저하되는 일 없이 계산 하는데 드는 시간과 memory footprint를 감소시켰다.
- Krizhevsky et al(2013)과 같은 더 작은 net을 사용하는 것은 심지어 light-weight GPU들에게서도 video-rate test-time에서 dense feature computation을 허용한다.

## Detailed Boundary Recovery: Fully-Connected Conditional Random Fields and Multi-Scale Prediction

## 추가적인 내용

배워야 할 keyword : ground truth label
