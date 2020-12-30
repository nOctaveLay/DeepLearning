
# U-Net

## Abstract
* 데이터 증강의 강력한 활용에 영향을 받는 network and training strategy
* 사용 가능한 작은 설명이 달려있는 data를 더 능률적으로 사용하기 위함
    * (approx.. 30 per application)
* Contracting path와 symmetric expanding path로 구성
    * Contracting path : context를 capture함
    * Symmetric expanding path : 정확한 localization을 가능하게 해줌
* 이런 network는 매우 적은 수의 이미지로부터 end-to-end(종단 간 종단 연결)로 segmentation을 학습
* 이런 네트워크는 ISBI에서 열렸던 electron microscopic stack 안에서의 신경 구조를 분리하는 대회에서 과거의 가장 훌륭했던 방법 (sliding-window convolutional network)를 압도함.
    * 512 x 512 이미지에서 빠른 network라는 것을 입증했다.
* 코드 : https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## Introduction
* 많은 visual task들에서 (특히 생물학 image processing에서) 요구되는 output은 localization을 반드시 포함해야 한다.
    * 즉, class label은 각각의 pixel에 할당된다고 가정된다.
* 생물의학에서, 훈련 이미지를 구하기 쉽지 않다.
    * 즉, 훈련 이미지를 줄이면서 학습 시킬 수 있어야 했다.
* Ciresan et al.은 sliding-window setup으로 network를 학습시켰다.
    * 목적
        * input으로 준 pixel의 주위를 local region으로 제공
        * 각각의 pixel의 class label을 예측하기 위함.
    * 장점
        * 첫째로, 이 네트워크는 localize 할 수 있다.
        * 둘째로, patch라고 불리는 training data들은 training image들보다 훨씬 많다.
    * 단점
        * Network가 매우 느리다.
        * 각각의 path를 위해 network가 독립적으로 실행되어야 한다.
        * Overlapping patch들 때문에 학습하는데 있어 많은 장애물들이 있다.
        * Localization accuracy와 context의 사용 사이에는 trade-off가 있다.
            * 더 큰 patch들은 localization accuracy를 줄이기 위해서 더 많은 max polling layer들이 필요하다.
            * 작은 patch들은 network들이 더 적은 context만 볼 수 있다.
            * 대부분의 연구들은 다중 layer들로부터 온 feature들을 설명하는 classifier를 제공했다.
            * 그렇지만, 좋은 localization과 좋은 use of context가 동시에 필요하다.
* U-net은 fully convolution network를 수정하거나 확장시켜서 만든 모델이다.
    * 이러면 좀 더 적은 수의 training image로 학습 시킬 수 있다.
    * 더 정확한 segmentation을 산출 할 수 있다.
    * [Fig 1]
* 이미지 segmentation에 있어 Fully convolution network의 Main idea
    * 연속된 layer에 의해 usual contracting network를 보충하는 것.
        * pooling operator는 upsampling operator로 대체
        * 이는 output의 resolution을 증가
    * localize하기 위해서 contracting path에서 온 high resolution features은 upsample된 output과 결합함
    * 이런 정보를 바탕으로 연속적인 convolution layer는 더 정확한 output을 모으는 것을 학습할 수 있다.
* Unet에서의 Main idea 수정
    * Upsampling part 
            * feature channel을 더 크게 만듬
                * network가 context information을 더 높은 resolution layer로 propagate 시키게 만듬
            * 결과적으로, expansive path는 contracting path와 symmetric하다.(대칭이다.) 
            * 따라서 u자형 모양이 나오게 된다.
        * Fully connected layer를 사용하지 않는다.
            * 단순히 각각의 convolution의 유효한 부분만 사용한다.
            * 즉, segmentation map은 입력 영상에서 full context만 사용 가능하게 하는 pixel만 담고 있다.
            * 이 방법은 overlap-tile 전략으로서, 임의의 커다란 이미지의 원활한 segmentation이 가능하다.
            * [Fig2]
            * 이미지의 경계 영역에 있는 픽셀을 예측하기 위해 입력 이미지를 미러링하여 누락된 컨텍스트를 추론한다.
            * 이런 타일 전략은 만약 타일 전략을 쓰지 않는다면 GPU 메모리 한계로 인해 resolution이 제한되기 때문에 큰 이미지에 network를 적용하기 위해서 중요하다.
        * 탄성 변형을 가능한 훈련 이미지에 적용함으로서 많은 양의 데이터 증가를 노린다.
            * 이건 network가 그런 이미지 망가짐에 적용되는 invariance를 학습시키도록 한다.
            * 주석 처리된 이미지 뭉치에서 이러한 이미지 변형을 볼 필요가 없도록 한다.
            * 이건 생체의학분류(biomedical segmentation)에서 중요하다.
                * 조직에서 생기는 일반적인 변형에 사용되는 이미지 망가짐과 실질적인 이미지 망가짐이 효율적으로 시뮬레이션되기 때문이다.
    * touching object를 같은 class로 분리
        * Fig 3.
        * Weighted loss 사용
            * touching cell들 사이에서 분리된 background label들이 큰 weight를 loss function에서 얻기 때문
* 다양한 biomedical segmentation 문제들에서 사용할 수 있을 것으로 기대

## Network Architecture
* Contracting path (left side) + Expansive path(right side)로 구성
* Contracting path
    * convolutional network의 일반적인 구조를 따른다.
    * 반복적인 두 개의 3x3 convolution (unpadded convolution)을 한다.
    * 각각은 ReLU(rectified linear unit)을 쓴다.
    * downsampling하기 위해서 stride 2의 2x2 max pooling을 한다. 
    * downsampling 단계마다, feature channel을 2배로 한다.
* Expansive path - 항상 ReLU를 씀
    * 2x2 convolution (up-convolution)으로 산출되는 feacure map의 upsampling
        * 이 convolution은 feature channels의 수를 반으로 쪼갠다.
    * contracting path에서 적당히 잘라진 feature map의 concatenation
        * cropping은 모든 convolution 안에서 가장자리 pixel들의 loss때문에 중요하다.
    * 각각 두 개의 3x3 컨볼루션
* 마지막 layer에서는 각각의 64 component feature vector를 원하는 class number로 매핑시켜주는 데 1 x 1 convolution을 사용한다.
* input tile size가 x축이든 y축이든 2x2 max-pooling operation이 되도록 하는 게 중요하다.  

## Training
* input image와 연관된 segmentation map은 network 학습에 사용
    * stochastic gradient descent implementation
* output image는 일정한 가장자리 width를 가지고 있는 input 보다 작다.
    * unpadded convolution 때문
* large batch size 위에 있는 large input tile을 씀
    * overhead를 minimize하고, GPU memory의 사용을 최대화하기 위함
    * 작은 이미지에는 작은 batch를 씀
* momentum을 크게 함 (0.99)
    * 과거에 보였던 training sample들의 많은 수가 현재의 최적화 step에서 update를 결정하도록 하기 위함.
* energy function
    * cross entropy loss function과 결합된 최종 feature map을 넘어선 pixel-wise soft-max에 의해 계산
    * [수식]
* cross entropy
    * [수식]
* Pre compute weight map
    * 각각의 ground truth segmentation이 training data set으로 특정 class에서 픽셀의 다른 빈도를 보상하기 위해.
    * 각각의 ground truth segmentation이 network가 작은 seperation border를 학습하도록 강요한다.
* Seperation border
    * morphological operation으로 계산된다.
* weight가 처음부터 좋은 게 좋다.
    * 그렇지 않으면, 다른 부분은 절대 기여하지 않는 반면에 network의 일부분이 과도한 activation을 준다.
    * 이상적으로, 처음 weight가 적용될 수 있다. 그래서 각각의 network 안에 있는 feature map은 근사적으로 unit variance를 갖는다.
    * architecture와 같이 있는 network를 위해서 (convolution과 ReLU layer들을 대체하면서) 표준 편차를 가지는 Gaussian distribution으로 부터 initial weight를 그림으로서 얻을 수 있다.

## Data Augmentation
* network에게 desired invariance와 robustness properties를 가르치는 것은 데이터 증가에 필수적이다.
    * 오직 아주 소수의 training sample만 가능할 때.
* microscopical image들의 경우, 우리는 회색값 변동과 변형에 대한 robustness 뿐만 아니라 shift, rotataion invariance도 필요하다.
    * 특히, 학습 샘플들의 랜덤 탄성 변형은 매우 적은 annotated image로 segmentation network학습에 대한 핵심 concept인거 같아 보인다.
* smooth deformation을 생성
    * random displacement vector 생성 
    * 거친 3 x 3 격자 이용
    * 그 배치는 10pixel의 표준 편차로 Gaussian distribution에서 sample되었다.
* 픽셀 단위의 배치는 bicubic interpolation을 이용해 계산되었다.
* contracting path에 있는 end에서 Drop out layer들은 더 많은 명시적인 데이터 증가를 수행했다.


## Experiments
* 3개의 다른 segmentation 의 application을 설명한다.
* 첫 번째 task는 electron microscopic recoding에서 neuronal structure를 segmentation 하는 것이다.
    * Training data = 30개의 이미지 (512 x 512)
    * 각각의 이미지는 cells(white) 와 membranes(black)의 cell을 위한 corresponding fully annotated ground truth segmentation map에서 왔다.
* evaluation은 예측되는 membrane probability map을 organizer에게 보냄으로서 얻어진다.
    * Evaluation은 맵을 10개의 다른 레벨과 "warping error", "Rand error", "pixel error"에 대한 계산으로 thresholding 함으로써 완료된다.
* u-net (평균 7번의 input data의 rotated version을 가진다.) 은 사전/후처리 없이 0.0003529 정도의 warping error, 0.0382의 rand error를 얻을 수 있다.
* 이건 sliding window convolutional network result보다 훨씬 더 좋다.
