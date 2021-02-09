# Rich feature hierarchies for accurate object detection and semantic segmentation

Tech report (v5)
Ross Girshick Jeff Donahue Trevor Darrell Jitendra Malik
UC Berkeley

- link : [here](https://arxiv.org/abs/1311.2524)
- code : [here](http://www.cs.berkeley.edu/˜rbg/rcnn.)

## Abstract

- 2개의 key insights

1. high-capacity convolutional neural networks (CNNS)을 bottom-up region proposal에 적용

   - object들을 localize하고 segment하기 위함

2. label이 붙여진 training data가 부족할 때, 보조적인 task를 위해 지도 선 학습 (정답을 알려주며 학습하는 것을 우선 하는 것)은 이전에 domain-specific fine tuning을 할 때 중요한 performance boost를 산출

- CNN과 region proposal을 결합하는 것을 R-CNN이라고 부르기로 함 -> Regions with CNN features.

- R-CNN과 OverFeat에 대한 비교

## Introduction

### visual recognition task

과거에는 SIFT와 HOG를 이용한 방법을 많이 했음

하지만 PASCAL VOC 물체 검출을 살펴봤을 때, 2010-2012년 동안 발전은 더뎠다.

(앙상블 시스템을 구축하는 것과 성공적인 방법에 대한 작은 변화를 채택하는 것으로 얻어진 작은 이득들만 있었다.)

SIFT와 HOG는 블록 방향 히스토그램이다.

블록 방향 히스토그램 : 대략적으로 V1에 있는 복잡한 세포들과 결합시킬 수 있는 표현이다.

V1 :  영장류의 시각 경로에서 첫번째 대뇌 피질 부위.

### 인지는 downstream에 있는 여러 단계로 일어난다.

이는 더 정보가 많은 feature를 계산하기 위해 계층적이고, 다중 스테이지 프로세스가 있을 것이라는 생각에 이르렀다.

### Fukushima의 neocognitron [19]

패턴 인식에서, 생물학적으로 영감을 받은 계층적이고 이동에 불변한 모델이다.

계층적이고 많은 스테이지가 있을 것이다라는 가정에서 시작한 과정의 앞선 시도였다.

하지만 이는 supervised training algorithm(지도 학습 알고리즘)이 부족했다.


이런 알고리즘이 Rumelhart et al.[33]에 의해 개발된 후, LeCun et al.[26]은 back-propagation 쪽으로 통하는 stochastic gradient descent가 CNN(convolutional neural network)를 학습하는데 효과적이라는 것을 알았다.

### CNN
neocognitron을 확장하는 모델의 class

CNN은 1990년대에 많이 사용되었으나, support vector machine의 부상으로 인해 주류에서 밀려나고 말았다.

2012년, Krizhevsky et al [25]는 대체로 더 좋은 이미지 분류의 정확도를 보여줌으로서 CNN에 대한 관심을 다시 일으켰다.

### ImageNet
주요 이슈 : ImageNet을 사용한 CNN 분류 결과가 어느 정도까지 object detection result를 일반화 할수 있을까?

이미지 분류와 물체 탐지에 대한 gap에 다리를 둠으로서 이를 해결했다.

### 방향성
처음에, 이 논문은 CNN이 어떻게 PASCAL VOC라는 주제에 대해 object detection을 하는 과제에서 더 높은 성능을 낼 수 있는지에 대해 HOG와 더 유사한 특징을 기반으로 한 system과 비교하여 다룰 것이다.

이를 설명하고, deep learning으로 localizing object하는 방법과 high-capacity model을 오직 작은 양의 annotation detection 데이터를 가지고 트레이닝 하는 것을 설명할 것이다.

### localizing
이미지 분류와 다르게, 이미지 탐색은 물체를 localizing하는 것이 필요하다.

#### 첫 번째 접근 : localization을 regression problem으로 프레임을 씌워보자.

하지만, Szegedy에서의 연구는 이러한 전략이 현실에서는 잘 되지 않았다고 말했다.

대안책으로 sliding-window detector를 개발했다.

CNN은 특히 object category에 한정해서 (예를들면 face나 pedestrian으로서) 적어도 20년동안 이러한 방식으로 사용되었다. 

높은 공간 resolution을 유지하기 위해서, 이런 CNN들은 2개의 convolutional layer와 1개의 pooling layer만 가져야 했다.


sliding-window 접근을 적용하는 것을 고려했다. 

하지만, 5개의 convolutional layer를 갖고 있는 R-CNN 네트워크의 높은 곳에 있는 unit들은 input image 안에서 매우 큰 receptive field(195 x 195)와 stride(32 x 32)를 들을 가진다.

이것은 sliding-window 패러다임 안에서 정확한 localization을 만든다.

#### 두 번째 접근 : regions을 사용한 인식 ("recognition using resion") 패러다임 안에서 작동시키는 것
object detection(object를 판단하는 것)과 semantic segmentation(장면을 완벽하게 분류하는 것)에서 성공적이었다.

테스트 때, 
input image를 위해서 2000개의 카테고리에 독립적인 region proposal들을 생성했다.

CNN을 사용해서 각각의 proposal에서 fixed-length feature vector를 추출한다.

그리고 각각의 region을 category에 특징적인 linear SVM으로 분류한다.


각 region proposal마다 고정된 사이즈의 CNN input을 계산하기 위해서 단순한 테크닉을 사용한다.
Figure 1은 method에 대한 개략적인 이해를 보여주고, 결과물의 일부를 highlight한다.
우리의 시스템이 CNN과 region proposal을 결합하기 때문에, 우리는 이 방법은 R-CNN : regions with CNN feature라고 이름 붙였다.

### update version
이 논문의 업데이트 버전에서, 
R-CNN을 200개의 class ILSVRC2014 detection dataset에서 돌린 최근의 제안된 OverFeat detection system과 R-CNN의 직접적인 비교분석을 제공한다.
detection : OverFeat < R-CNN

### detection에서 직면된 두 번째 문제 : labeled data가 부족하고, 현재 사용가능한 데이터 양이 부족:
conventional solution : supervised fine-tuning 후 **unsupervised** pre-training (비지도 사전 학습) 
하지만 이 논문에서는 domain-specific fine-tuning (PASCAL에 적용) 후 **supervised** pre-training (지도 사전 학습) 
이 방식이 데이터가 부족할 때 high-capacity CNN을 학습시키는 것을 위해서는 효율적인 패러다임이다.

이 실험에서, detection을 위한 fine-tuning은 mAP performance를 8%나 상승시켰다.
fine tuning 후, HOG를 기본으로 한 deformable part model (DPM)이 33퍼의 mAP를 얻은 것에 비해, VOC에서 mAP를 54퍼나 얻었다.

이 논문은 장면 분류, fine-grained sub-categorization, domain adaptation 같은 몇 개의 recognition 과제에 훌륭한 성과를 드러내면서 Krizhevsky의 CNN이 blackbox feature extractor로 사용될 수 있다는 것을 보여준 Donahue et al의 동시대의 다른 모델들을 지적한다.

### efficient
유일하게 클래스에 특별한 계산은 합리적으로 작은 행렬 벡터 곱과 greedy non-maximum suppression으로 구성되어있다.
이런 계산적인 속성은 모든 범주에 의해서 공유되며 이전에 사용된 region feature보다 크기가 더 낮은 특성에서 따온 것이다.

### 이해
우리의 접근의 실패한 모드를 이해하는 것은 이것을 향상시키는 데 중요하다. 
Hoiem et al의 detection analysis tool로 report를 결론내릴 것이다.
이러한 분석의 즉각적인 결과로서, simple bounding box regression method가 mislocalization을 줄이는 것을 중요하게 설명할 것이다.
mislocalization은 error mode에서 가장 많이 나온다.
technical detail을 개발하기 전에, R-CNN이 region에서 작동하기 때문에 이게 semantic segmentation으로 확장되는 것은 자연스러운 일이다.
자그마한 수정으로, 경쟁적인 결과물을 얻을 수 있었다.

## Object detection with R-CNN
object detection system은 3개의 모듈로 구성된다.
1. category-independent region proposals - detecor에 사용 가능한 detection 후보 집합
2. large convolutional neural network - 고정된 길이의 특징 벡터를 추출
3. a set of class specific linear SVMs

### Module design
**Region proposals** - 최근 연구의 다양성은 category에 독립적인 region proposal을 만들기 위해 method를 제공한다.

**Feature extraction** - 각각의 region proposal에서 Krizhevsky et al.에 의해 묘사된 CNN의 Caffe를 사용한 구현을 사용해서 4096개의 차원을 가진 feature vector를 추출한다.
 
Feature은 5개의 convolutional layer와 2개의 fully connected layer를 통해 mean-subtracted 227 x 227 RGB 이미지를 forward propagating하는 것에 의해 계산된다.

더 자세한 언급 [24,25]번 논문 참조

region proposal을 위한 feature를 계산하기 위해서, 우리는 처음에 region 안에 있는 이미지 데이터를 CNN에 compatible한 형태로 바꿀 필요가 있다.

CNN은 227 x 277 픽셀 사이즈로 input 픽셀 사이즈가 고정되어있다.

임의의 형태를 한 region의 가능한 많은 변형 중에서, 가장 단순한 방법으로 최적화시켰다.

size와 후보 지역(candidate region)의 가로 세로 비율 (aspect ratio)에도 불구하고, 우리는 모든 pixel을 요구되는 사이즈에 매우 tight한 bounding box로 감쌌다.

감싸는 것에 앞서, 우리는 tight bounding box를 팽창시킨다.

감싸는 사이즈에, 원래의 box 주위에 정확히 감싸진 이미지 context의 p 픽셀들이 있다. (이 논문에서 p = 16이다)
Fig 2 = random sampling of warped training regions

### Test-time detection
2000개에 가까운 region proposal을 추출하기 위해서 test image에서 selective search를 돌렸다.

(이 논문에서는 모든 실험에서 selective search의 fast mode를 사용했다.)

각각의 proposal을 감싸고, feature를 계산하기 위해서 CNN을 통해서 forward propagate를 시켰다.

각각의 클래스를 위해서 각각의 추출된 feature vector에 그런 클래스를 위해 학습된 SVM을 사용해서 점수를 매겼다.

이미지 안에서 모든 점수가 매겨진 region이 주어지면, greedy non-maximum suppression을 (각각의 클래스에 독립적으로) 적용했다.

greedy non-maximum - 우리는 만약 학습된 임계값보다 더 큰 덤수를 획득한 선택된 영역을 가지는 intersection-over-union(IoU) overlap을 가지면 region을 거절하는 것

**Run-time analysis**
두개의 특성이 detection을 효과적으로 만든다.
1. 모든 CNN parameter들이 모든 카테고리를 경유하여 공유된다.
    * 결과 : region proposal이나 feature를 계산하는데 걸린 시간이 모든 class를 거쳐 분할 상환된다.(amortize)
    * 유일한 클래스에 특화된 계산은 feature와 SVM weigth, non-maximum suppression 사이의 dot product
    * image를 위한 모든 dot product는 single matrix-matrix product로 일괄처리 (batch)된다.
    * feature matrix는 일반적으로 2000 x 4096
    * SVM weight matrix는 4096 x N (N은 class의 갯수)

1. CNN에 의해 계산된 feature vector가 low-demensional하다.
    * 위의 분석은 R-CNN이 hashing 같은 기술로 근사화 해야 하는 것 없이 수많은 object의 class를 증가시킬 수 있다는 것을 보여준다.
    * 100k class가 있다 할지라도, 오직 10초만 걸린다. 이건 단순히 region proposal이나 공유된 feature을 사용한다고 얻어지는 결과가 아니다.
    * UVA system은, high-dimensional feature 때문에  크기를 더 줄이는 2개의 순서가 있다. 이는 134GB 메모리를 필요로 하고, 저장하는데 100K linear predictor를 요구한다.
    * 반면에 R-CNN은 1.5GB 메모리만 필요로 한다.
    * 단, 다른 일반적인 접근하고 비교되었을 때.
    * bag-of-visual-word로 암호화된 spatial pyramid같은게 있다.
이런 UVA detection system에서 사용되는 이런 feature들은 논문에서 나온 것들보다 크기를 더 키우는 두개의 order가 있다.

scalable detection에서 DPM과 hashing을 사용하는 Dean et al의 최근 연구와 R-CNN을 대조해보는 것은 흥미롭다.

Dean et al. 연구는 10k개의 distractor class들을 소개할 때 VOC 2007에서 16%의 mAP와 이미지를 읽는데 5분의 시간이 걸린다.

우리의 연구로는, 10k의 detector가 CPU하나에서 1분만에 실행 가능하다. 어떠한 approximation도 만들어지지 않기 때문에 mAP는 89퍼로 유지한다.

## Training
**Supervised pre-training**
image-level annotation만 사용해서 차별적으로 큰 임의적인 dataset에서 CNN을 미리 훈련시켰다.

선 학습은 open source Caffe CNN 라이브러리를 사용해서 수행되었다.

짧게, 이 논문의 CNN은 Krizhevsky et al의 성능에서 top-1 error rate가 2.2 percent point 높을 정도로 거의 일치한다. 이런 불일치는 학습 과정이 단순화됬기 때문이다.

**Domain-specific fine-tuning**
CNN을 새로운 과제(detection)과 새로운 도메인(warped proposal window)에 적용시킴으로서, 오직 warped region proposal만을 사용해서 CNN parameter에 대해 stochastic gradient descent (SGD) training을 지속한다.

CNN의 ImageNet에 특화된 100가지 방법 classification layer를 랜덤하게 초기화 된 (N+1) 방법 classification layer로 대체한 것과는 다르게, (N은 object class의 수에 background로 1개를 더한 갯수) CNN 구조는 변하지 않는다.

    1. VOC에서, N = 20이다.
    2. ILSVRC2013에서, N = 200이다.

우리는 모든 region proposal을 ground-truth box는 positive들로, box의 class나 나머지는 negative로 다루면서 0.5 IoU overlap 이상으로 다룬다.

SGD를 0.001의 learning rate로 시작한다. (처음 선행 학습률의 1/10정도) 이건 fine-tuning이 초기화를 방해하는 것이 아니라 progress를 만들 수 있도록 한다. 
 
각각의 SGD iteration 동안, 균일하게 32개의 positive window (모든 클래스에 대함)들과 128 사이즈의 mini batch를 구축할 96개의 백그라운드 윈도우를 샘플링했다.

우리는 sampling을 positive window로 편향했다. 이는 백그라운드와 구분되기엔 극단적으로 어려웠기 때문이다.

**Object category classifier**
차를 detect하는데 있어 binary classifier로 훈련하는 것을 생각해보자.

차를 둘러싼 tight한 이미지 region은 positive example일 것이다.

유사하게, background region, 차랑 아무 관련도 없는 곳,은 negative example일 것이다.

좀 덜 명확한 것은 차를 부분적으로 감싸고 있는 지역은 어떻게 label할 것인가 이다.

이 문제를 region이 negative로 정해진 IoU overlap threshold를 가지고 풀려고 한다. 

overlap threshold, 0.3은 validation set인 {0,0.1, ... , 0.5} 안에서 grid search에 의해 선정된다. 

이런 threshold를 조심스럽게 선택하는 것은 중요하다. 0.5로 세팅하면, mAP가 5 point까지 줄어든다. 0으로 선택하면, mAP가 4 point까지 줄어든다.

positive example은 단순히 각각의 클래스의 ground-truth bounding box로 정의된다.

일단 feature은 추출되고, training label은 적용된다. SVM을 클래스마다 최적화한다.

training data가 메모리에 맞추기에 너무 크기 때문에, 우리는 standard hard negative mining method를 적용시켰다[17,37].

Hard negative mining 은 빠르게 수렴된다. 그리고 실제로 모든 이미지에 오직 하나의 pass만 거치면 mAP는 증가를 멈춘다.

Appendix B에서 왜 positive와 negative example이 SVM에 비해 fine-tuning에서 다르게 정의되는 지를 논의할 것이다. 우리는 훈련 하는데 단순히 fine-tuned CNN의 최종 softmax layer에서 나온 output을 사용하는 것 보다 detection SVM으로 훈련시키는 것에 대한 것을 포함한 trade off를 논의할 것이다.

### Results on PASCAL VOC 2010-12

PASCAL VOC의 가장 좋은 시도를 따르면 [15], 우리는 모든 디자인 결정과 VOC 2007 dataset에 있는 hyperparameter를 확인해야한다.

VOC 2010-12 dataset의 최종 결과물에서, VOC 2012 훈련에 있는 CNN을 잘 조정했다.
그리고 VOC2012 trainval을 구동하는 SVM에 있는 detection을 최적화했다.

테스트 결과를 evaluation server에 오직 한번만 두개의 major algorithm variant를 각각 돌리는 데 썼다. (with and without bounding-box regression)

Table 1은 VOC 2010의 완벽한 결과를 보여준다.

우리는 SegDPM을 포함한 4개의 강한 baseline에 대항하여 우리의 방법을 비교한다.

SegDPM은 DPM detector와 semantic segmentation system의 output과 결합한 것이다.

우리는 추가적인 inter detector context와 image-classifier rescoring을 사용한다.

가장 적절한 비교는 우리의 시스템이 같은 region proposal 알고리즘을 사용하기 때문에, Uijlings et al로부터 온 UVA system[39]과 비교하는 것이다.

region을 분류하기 위해서, UVA는 4개의 공간 피라미드 (spatial pyramid)를 만들었다. 그리고 그것을 각각의 벡터가 4000개의 단어로 되어있는 codebook으로 양자화 되어있는 densely sampled SIFT, Extended OpponentSIFT, 그리고 RGB-SIFT descriptor로 만들었다. 

Classification은 histogram intersection kernal SVM으로 작동된다.

대부분의 multi-feature와 non-linear kernel SVM 접근을 대조했을 때, 우리는 35.1%에서 53.7%까지 mAP에서 커다란 향상을 얻을 수 있었다. 

우리의 방법은 VOC 2011/12 테스트에서 유사한 성능을 얻는다.

### Results on ILSVRC2013 detection

200개 클래스로 되어있는 ILSVRC2013 detection dataset에서 PASCAL VOC에서 돌렸던 것과 같은 system hyperparameter R-CNN을 돌렸다. 우리는 ILSVRC2013 evaluation server에서 2번밖에 안돌린 테스트 결과를 제출하는 것에 대해 같은 protocol을 사용했다. 한 번은 bounding box regression을 사용하고, 한 번은 사용하지 않았다.

Figure 3은 R-CNN과 ILSVRC 2013 대회에 나왔던 다른 참가자들을 비교한다. 그리고 다음 경쟁자 OverFeat과의 비교도 보여준다. R-CNN은 mAP를 31.4% 얻었다. 이것은 2위를 차지한 OverFeat가 24.3%를 차지한 것보다 훨씬 앞섰다.

class 당 AP distribution을 알아보기 위해서, box plot이 제안되고 AP 당 class table은 Table 8에 있는 논문의 끝을 다른다. 대부분의 경쟁자들은 (OverFeat, NEC-MU, UvA-Euvision, Toronto A, UIUC-IFP) 그들이 어떻게 CNN이 object detection에 적용될 수 있는지를 보여주는 중요한 뉘앙스를 보여주면서 convolutional neural network를 사용했다.  이는 매우 다른 결과를 가져왔다.

Section 4에서, 우리는 ILSVRC2013 detection dataset에 대한 개요를 줄 것이고, R-CNN을 사용하면서 우리가 내린 선택에 대해 detail을 제공할 것이다.



## 추가 조사 필요한 것
* mAP
* UVA system
* non-maximum suppression
* intersection-over-union
* stochastic gradient descent (SGD)