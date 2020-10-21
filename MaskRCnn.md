
# Mask R-CNN

https://arxiv.org/pdf/1703.06870.pdf Facebook Research

## Mask R-CNN 요약정리
### 기본 정리

* COCO suite of challenge 의 3개의 종목에서 우수한 성적을 거둔 model이다.
 
* 기본적으로 R-CNN과 Image Segmentation에 초점을 두고 있다.

* Faster R-CNN을 개선한 방법이다. 이미지 detection을 잘한다.

* 훈련시키기 간단하고, Faster R-CNN에 작은 overhead만 첨가된다.

* Faster R-CNN에 한 가지의 branch만 더 첨가했다. 이것은 mask predict branch라고 한다.

* 다른 task들을 일반화 하기 쉽다.

* 코드 : https://github.com/facebookresearch/Detectron

* R-CNN : Bounding box object detection 하기 위함

* Image segmentation - Instance first strategy

### Mask R-CNN의 원리

* Faster R-CNN (class label, bounding-box offset) + object mask

* 2개의 스테이지로 구성된다.
    * RPN
    * Class and box offset을 예측하는 것과 병행하면서 (parallel) 각각의 RoI를 위해 binary mask를 산출한다. 


## Mask R-CNN 전체 논문 해석
>R-CNN -> Fast R-CNN -> Faster R-CNN -> Mask R-CNN
>
>= Instance segmentation을 위한 방법. Fast R-CNN을 개선한 방법이다.
>
>(Fast R-CNN : R-CNN을 개선한 방법. Ross Girshick의 Rich feature hierarchies for accurate object detection and semantic segmentation의 확장이다.)

### Abstract
* object instance segmentation에 대해 간단하고 유연하고 일반적인 framework를 제시
* 각각의 instance에 대해 높은 quality의 segmentation mask를 생성하면서 효과적으로 object에 있는 이미지를 detect함. 
* Mask R-CNN 이라고 부르는 이 방법은 Faster R-CNN에 object mask를 예측하는 branch를 첨가하고 기존의 bounding box recognition을 위해 존재하는 branch를 동시에 돌린다.
* Mask R-CNN은 train하기 쉬움, 그리고 Faster R-CNN에 대해 오직 작은 overhead만 더함.
    * train하기 간단하다 : 넓은 범위의 유연한 구조 설계가 가능하게 한다.
    * 작은 overhead만 첨가된다 : 빠른 시스템과 빠른 설명이 가능하게 한다.
* Mask R-CNN은 다른 task들을 일반화 하기 쉬움.
    * 예를 들면, 같은 프레임워크 내에서 우리에게 사람들의 포즈를 추정할 수 있도록 도와줌
* COCO suite of challenge의 3개의 종목(instance segmentation, bounding box object detection, person keypoint detection)에서 우수한 성적을 거뒀음.
* Bell과 whistle 없이, Mask R-CNN은 모든 존재하는 single model들보다 훨씬 더 우수한 성능을 보임.
* https://github.com/facebookresearch/Detectron << 코드

### Introduction 
* 목표 : instance segmentation을 위해 비교 가능한 framework를 개발하는 것이다.
* 고려한 사항
    * Image segmentation 

        * 문제 
            * 모든 object들의 correct detection이 필요함
            * 정확하게 각각의 instance들을 분리 해야 함.

        * 이 문제를 해결하기 위해선 전통적으로 썼던 Object detection과 관련된 computer vision task의 요소들을 결합해야 함.

        * Object detection의 목적 
            * 각각의 object들을 명확히 하고, 사용하는 bounding box를 각각 localize 하고, semantic segmentation 하는 것이다.

            * Semantic segmentation의 목적 : 
                * 각각의 pixel을 object instance들을 구별 짓는 것 없이 고정된 카테고리의 집합으로 명확히 하는 것이다. 

    * existing branch (Faster R-CNN) + **predict branch** 

        * predict branch : 각각의 Region of Interest(RoI)의 segmentation mask들을 예측하는 branch
        * classification 그리고 bounding box regression을 위한 branch 
        * existing branch와 predict branch를 동시에 첨가함으로서 Faster R-CNN 확장

    * Mask branch

        * pixel 단위로 segmentation mask를 예측하면서 각각의 RoI에 적용되는 자그마한 FCN(Fully convolution network)이다.
        * mask branch를 적절하게 설계하는 것은 좋은 결과를 위해 필수적이다.

    * RolAlign

        * Faster R-CNN은 network input과 output 사이에서 픽셀 대 픽셀 정렬을 위해 디자인되어있지 않다.
            * 이는 인스턴스 참석을 위한 de facto 핵심 연산인 RolPool이 feature extraction을 위한 거친 공간 정량화(coarse spatial quantization)을 수행하는 방법에 잘 드러나있다.

        * RolAlign - 픽셀 대 픽셀 정렬을 고려하기 위해서, 정확한 공간 위치를 신뢰성 있게 보존할 수 있는, 즉 단순하고, quantization-free layer(양자화에 자유로운 layer)를 제시했다.

            * RoIAlign은 보다 엄격한 localization matrics에서 훌륭한 성과를 거두면서 mask accuracy를 10% -> 50%로 상승시켰다. 
            * 마스크와 class 예측을 decouple하는 것이 필수적인 것을 밝혀냄 : class들 간에 competition없이 독립적으로 각각의 class를 위해서 binary mask를 예측. 그리고 카테고리를 예측하기 위해서 network의 RoI classification branch를 사용.
                * 대조적으로 FCNs는 pixel단위로 multi class categorization을 실행함.
                    * Multi class categorization이란? Segmentation과 classification을 묶는 것.
                    * FCNs는 instance segmentation에 낮은 성적을 보여줌.
        * 절제 실험에서 여러 기본 인스턴스 화를 평가하는데, 이를 통해 robustness를 입증하고 핵심 요인의 효과를 분석
    
* 성능
    * GPU에서 프레임당 200ms를 돌릴 수 있음.
    * COCO 예제를 training 했을 때 하나의 8 GPU machine으로 1-2일이 걸렸음.
    * 빠른 훈련과 테스트 속도가 프레임워크의 유연성과 정확성과 함께 instance segmentation에서의 미래 연구에 값어치가 있고 도움이 될 것이라고 생각한다.
    * COCO 키 포인트 데이터 집합에 대한 인간 포즈 추정 작업을 통해 프레임워크의 일반성을 보여준다.
        * 각각의 keypoint를 one-hot binary mask로 봄으로서, 작은 수정사항과 함께 Mask R-CNN은 instance-specific 포즈들을 탐지하는 데 적용될 수 있다.

### Related Work
#### R-CNN – Region-based CNN, 정확한 object detection을 위해 풍부한 feature hierarchies(특징 계층)을 만드는 방법. 
* Bounding box object detection에 대한 Region-based CNN 접근
    * 조정 가능한 candidate object 영역의 수를 고려
    * convolutional network를 각각의 RoI마다 독립적으로 평가
* R-CNN 확장 
   * fast speed와 더 나은 정확도를 위함
   * feature map에 RoI가 관여하는 것을 허가
* Faster R-CNN
    * 지역 제한 네트워크를 통해 attention mechanism을 학습함으로써 이 스트림을 발전시켰다.
    * 많은 follow-up improvement들에게 flexible 하고 robust하다.
    * 여러 benchmark에서 주도적인 framework이다.
* https://arxiv.org/abs/1311.2524

#### Instance Segmentation
* R-CNN의 효과에 따라, 많은 instance segmentation 연구들이 segment proposal에 기반을 두었다. 
    * 이전의 연구들은 bottom-up segment로 재 분류 되었다.
* DeepMask와 이에 연관된 연구 - Fast R-CNN으로 정의되는 segment candidate를 제안
    * Recognition(인식)을 하기 전에 먼저 segmentation(분리)을 한다. 
         * 느리고 덜 정확하다.
    * 이와 같이, Dai et al.은 segment proposal을 예측하는 complex multiple-stage cascade (복잡하고 다양한 연속 stage)를 제시했다. 
        * 이는 후에 classification을 한다.
    * 대신에, 우리의 method는 mask와 class label들의 **병행(parallel) 예측**에 기반을 두고 있다. 
        * 이것은 더 단순하고 더 유연하다.
    * 2가지 예측
        * 최근에 Li et al.은 segment proposal system과 object detection system을 fully convolutional instance segmentation (FCIS)를 위해서 결합시켰다. 
            * 일반적인 생각 : position sensitive output channel들의 집합을 fully convolutionally하게 예측하자.
            * 장점 : 이러한 채널들은 시스템을 빠르게 하면서 동시에 object class, box, mask들을 알아낸다.
            * 단점 : FCIS는 instance들을 overlapping 할 때 발생하는 system적인 error들도 금지하고, 따라서 가짜 edge들을 만들어 낸다.
                * 이건 instance들을 분리하는 것을 구조적으로 어렵게 만들었다.
        * 또 다른 solution들은 semantic segmentation의 성공에서 왔다.
            * pixel단위의 classification 결과로 시작해서, 이러한 방법들은 같은 카테고리의 pixel들을 다른 instance들로 자르는 시도를 했다. (segmentation-first strategy)
    * 이러한 방법들과는 다르게, Mask R-CNN은 instance-first strategy를 쓴다.

### Mask R-CNN
* Faster R-CNN은 각각의 후보 object에 2개의 output을 가지고 있다.
    * Class label과 bounding box offset이다.
* Faster R-CNN에 1개의 output을 더 추가한다. 
    * 이는 output이 object mask인 것이다.
    * 이 새로운 mask의 output은 class와 box output으로부터 떨어져있다.
    * 이는 좀 더 잘 맞는 object의 공간적인 layout의 추출을 필요로 한다.
* 픽셀 단위 배치와 함께 Fast/Faster R-CNN에서 추가된 Mask R-CNN의 핵심 element를 소개한다.
    #### Faster R-CNN: 
    * 2개의 스테이지 - RPN and RoIPool
        * Region Proposal Network (RPN) – candidate object bounding boxes를 제시한다.
        * 각각의 후보 box에서 특징 추출 using RoIPool – Fast R-CNN에서 핵심이다.
    * 이러한 특징들은 좀 더 빠른 inference를 위해 공유된다.
    #### Mask R-CNN : 똑같이 2개의 스테이지로 구성된다.
    * 2개의 스테이지 - RPN and binary mask
        * RPN (Faster R-CNN과 똑같음)
        * Class and box offset을 예측하는 것과 병행하면서 (parallel) 각각의 RoI를 위해 binary mask를 산출한다. 
           * 이는 classification이 원래 mask prediction에 의존하는 것과는 정반대의 시스템이다.
           * Bounding box classification과 regression을 병행으로 처리할 때에는 Fast R-CNN 방식을 따른다.)
* Training 하는 동안 각각의 sampled RoI에서 multi task loss를 다음과 같이 정의한다
    * L = L(class) + L(box) + L(mask)
        * L(class) = the classification loss
        * L(box) = the bounding-box loss
        * L(class) 와 L(box)는 identical하다.
        * Mask branch는 각각의 RoI만큼의 K m^2 dimensional output을 가지고 있다.
            * 이것은 m * m resolution을 가진 K binary mask로 encode 됨을 의미한다.
            * One for each of the K classes.
        * 우리가 각 픽셀마다 sigmoid 함수를 적용시키기 위해서, L(mask)는 평균적인 binary cross-entropy loss로 정의된다.
        * Ground truth class k와 연관된 RoI를 위해서, L(mask)는 오직 k-th mask로 정의된다. (다른 마스크는 loss에 기여하지 못한다.)
        * L(mask)의 이런 정의는 class들과 경쟁하는 일 없이 network에게 mask를 모든 class를 위해 생성하도록 허락해준다.
            * Output mask를 선택하기 위해 사용되는 class label을 예측하기 위해서 전용 classification branch에 의존한다.
            * 이건 mask와 class prediction을 분리시킨다.
        * 이건 FCNs에서 픽셀 별로 multinomial cross-entropy loss와 softmax를 사용하는 semantic segmentation에 적용할 때랑 다른 방식이다. (이런 방식들은 class를 가로지르는 mask를 완성시킨다. )
            * Mask R-CNN은 sigmoid와 binary loss를 쓴다 할지라도 class를 가로지르는 mask를 완성시키지 않는다.
* Mask Representation 
    * Mask는 input object의 공간적인 layout을 해석한다.
        * 불가피하게 fully-connected layers(fc)에 의해서 short output vector로 축소되는 Class label 또는 box offset과는 다르게, 마스크의 공간적인 구조를 추출하는 것이 convolution에 의해 제공되는 픽셀 단위의 상호 반응성에 의해 자연스럽게 설명된다.
        * Mask 예측을 위해 fc layer를 resort 하는 기존의 방법과는 다르게, mask R-CNN의 fully convolutional representation은 거의 parameter가 필요하지 않고, 실험에서 제시된 것처럼 더 정확하다
    * 특히, 우리는 FCN을 사용해서 각각의 RoI로부터 m x m mask를 예측할 수 있다. ( J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015)
        * 이건 mask branch에 있는 각각의 layer가 공간적인 layout을 공간 차원을 부족하게 만드는 vector representation으로 바뀌는 일 없이 정확한 m x m object의 공간적인 layout을 유지하는 것을 허용해 준다. 
        * 이런 픽셀과 픽셀의 행동은 이들 자체가 명확한 픽셀 단위의 공간 상호 응답성을 보존하기 위해 신뢰성 있게 잘 할당된, 작은 feature map인 RoI 특징을 필요로 한다.
        * mask 예측에서 핵심적인 역할을 하는 RoIalign layer 만듬 
            * RoIAlign = RoIPool을 개선한 RoIAlign
            * RoIPool은 각각의 RoI에서 small feature map을 추출하기 위한 standard operation이었다.
                 * RoIPool은 처음에 floating number RoI를 피처맵의 개별적인 세분화로 양자화 한다. 
                 * 이 양자화된 RoI는 그들 스스로가 양자화된 공간적인 bin으로 세분화 된다.
                      * 양자화 = 숫자를 표기하는 bit의 수를 줄이는 것.
                      * 양자화는 예를 들면 연속적인 변수 x가 [x/16]을 계산할 때 수행된다.
                      * 이 때 16은 feature map이고, []는 rounding이다.
                 * 마침내 각각의 bin에 의해 커버된 feature value들은 집합으로 묶여진다. (max- pooling에 의함)
                 * 이러한 양자화는 RoI와 추출된 특징 사이에서 잘못된 정렬을 보여준다.
                 * 이건 자그마한 변화에 robust한 impact classification이 아니다. 
                 * 이것은 pixel에 정확한 mask를 예측하는 것에 커다란 부정적인 영향을 끼친다.
                 * 이를 설명하기 위해서, input에서 추출된 특징들을 적절하게 배치하면서 RoIPool의 harsh quantization을 다시 옮기는 RoIAlign layer를 제시한다.
           * 제안된 변화는 단순하다 : 우리는 RoI 경계나 bins에 대해 어떠한 양자화도 하지 않는다. (예를 들어, [x/16]보다는 x/16을 쓴다.) 
               * 4개의 규격화된 샘플로 된 location에서 input feature들의 정확한 value를 계산하기 위해 Bilinear interpolation을 쓴다. 그리고 결과를 모은다.  
               * 결과는 정확한 sampling location이나 얼마나 많은 point들이 sample지에 대해 민감하지 않다.
               * 4.2에서 보여주듯, RoIAlign은 많은 향상을 이룩했다. 
           * RoIWarp와의 비교 -> 이는 **배치의 중요성**을 보여준다.
               * RoIWarp는 잘못된 배치로 인해 정보가 소실되는 것을 간과했다.
               * RoIPool처럼 RoI를 양자화함으로서 실행시켰다.
               * RoIWarp는 bilinear resampling이 적용됬다. 
               * 하지만 RoI를 양자화 시킨 후 실행시켰기 때문에, RoIPool과 동등하게 작동했다.
          
* Network Architecture
    * 이거에 대한 접근을 듣기 전에, Mask R-CNN을 다양한 구조에서 인스턴스화 시킬 필요가 있다.
    * 명백하게, The convolutional backbone architecture와 network head를 분리시켰다.
        * Convolutional backbone architecture
            * Convolutional backbone architecture는 전체적인 이미지의 특징 추출에 사용된다.
                * Backbone architecture는 network depth feature를 사용하는 것으로 정의한다.
                * Depth 50 또는 101 layer를 갖고 있는 ResNet과 ResNeXt 네트워크로 주로 구성된다.
                    * ResNet 네트워크
                        * ResNet을 가지고 있는 Faster R-CNN의 기본적인 구현은 4개의 stage로 되어있는 최종적인 convolutional layer로부터 특징을 추출한다. (이를 C4라고 부른다.)
                        * ResNet-50을 가지고 있는 backbone은 ResNet-50-C4라고 불린다. 이는 일반적으로 ResNet에 사용된다.
                * Lin et al.에 의해 주장된 훨씬 더 효율적인 backbone은 Feature Pyramid Network(FPN)이라고 불린다. 
                    * FPN은 lateral connection으로 single scale input으로부터 network feature pyramid를 짓기 위해 top-down architecture를 사용한다.
                    * FPN backbone을 가지고 있는 Faster R-CNN은 그들의 크기에 따라서 다른 레벨의 feature pyramid에서 RoI 특징들을 추출한다. 
                    * 달리 말하자면 일반적인 ResNet과 유사한 접근을 가진다.
               * 특징 추출에 있어서 Mask R-CNN에 ResNet-FPN backbone을 사용하는 것은 정확도와 speed에 있어서 훌륭한 결과물을 산출했다. 
        * Network head는 bounding box recognition과 각 RoI에 독립적으로 적용되는 mask prediction에 사용된다. 
             * ResNet과 FPN 논문에서 온 Faster R-CNN box head를 확장한다.
             * ResNet-C4 backbone은 ResNet의 compute-intensive한 5개의 stage를 포함한다.
             * FPN에서, backbone은 이미 res5를 포함하고 있다. 
                 * 더 적은 filter를 사용하는 더 효율적인 head를 위해 res5를 허용한다.
        * Mask branch는 직관적인 구조를 가지고 있다.
            * 더 복잡한 디자인은 성능을 향상시킬 수도 있겠지만 우리의 관심사는 아니다.

## Implementation Details
* Hyper-parameters를 Fast/Faster R-CNN work로 설정했다.
    * 이렇게 결정할지라도, instance segmentation system은 robust함을 알아냈다.
* 훈련
    * Fast R-CNN에 있을 때에는 RoI가 positive라고 생각됨 (만약 RoI가 적어도 0.5의 ground-truth box를 가지고 있는 IoU를 가지고 있다면) 그렇지 않다면 negative로 간주
        * L(mask)는 positive RoI를 가짐
        * Mask target은 RoI와 RoI와 연관된 ground-truth mask의 교차지점이다.
    * Image-centric training을 적용시킴
        * 800 픽셀로 이미지가 resize 될 수 있음
        * 각각의 mini batch는 GPU당 이미지 2개로 잡음
        * 이미지는 positive : negative = 1:3인 N sampled RoI를 가지고 있음
        * N은 64이거나 (C4 backbone에 의함) 512이다. (FPN에 의함)
        * 이 논문에서는 160k개의 iteration을 위해 8개의 GPU를 이용 (그러므로 mini batch size가 16) 
            * 학습률은 0.02로 120K번의 iteration에서 10개가 감소했다.
        * 0.0001의 weight decay를 사용했고, 0.9의 모멘텀을 사용했다.
        * ResNeXt에서, GPU당 1개의 이미지를 사용했고, 같은 수의 iteration을 사용 
            * starting learning rate는 0.01이었다.
        * RPN anchor는 5 scale과 3 aspect ratio로 늘렸다. 
    * 쉽게 모델이나 feature를 제거하기 위해서, RPN은 각각 훈련된다. 
        * 명시되지 않는다면 Mask R-NN과 같은 특징을 공유하지도 않는다. 
    * RPN과 Mask R-CNN은 같은 backbone을 가지고 있다.
        * 따라서, 공유 가능하다.

* Inference (추론)
    * Test 때 C4 backbone에서 제시된 숫자는 300이었다. 그리고 FPN에서 제시된 숫자는 1000이었다.
    * Non maximum suppression으로 결론이 나는 box prediction branch를 넣었다.
    * 이 mask branch는 가장 높은 스코어를 매긴 100개의 detection box에 적용되었다.
        * Mask R-CNN은 Faster R-NN보다 작은 오버헤드만 더한다는 것을 명심해라

    * 이게 훈련에 사용된 parallel computation과는 다른 길이다.
        * 그렇지만 이것은 간섭의 스피드를 증가시키고, 정확도를 향상시킨다. 
        * 더 적지만 더 정확한 RoI를 사용했을 시에만 해당된다.
    * Mask branch는 RoI마다 K마스크를 예측할 수 있다. 그러나 여기선 k번째 마스크를 사용했다. 
        * k는 classification branch에 의해 예측되는 class이다.
    * M x m floating number mask output은 다음에 RoI 사이즈로 리사이즈 된다. 그리고 0.5 threshold로 이분화된다.
