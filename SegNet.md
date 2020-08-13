
# SegNet Review
- SegNet : A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
- https://arxiv.org/pdf/1511.00561.pdf

# Abstract
- pixel-wise classification layer를 위한 Encoder network와 Decoder Network로 구성
- Encoder Network
  - VGG16에서 13개의 convolutional layer에 topologically identical하다. (위상학적으로 동일하다)
- Decoder Network
  - pixel-wise classification을 위해 low resolution encoder feature map을 full input resolution feature map으로 매핑시키는 과정
- 특이사항
  - decoder가 segnet의 더 낮은 resolution input feature map을 upsampling 하는 게 특이점이다.
  - decoder가 corresponding encoder가 non-linear upsampling을 수행하기 위해 max-pooling setp 안에서 계산되는 pooling indices를 쓴다.
    - 이는 학습이 upsample할 필요를 없앤다.
  - 이 upsampled map은 희소행렬이고, dense feature map을 생산하기 위해 훈련 가능한 필터와 convolve되어 있다. 
 - 동기 : scene가 application을 이해하는 것에서 출발했다.
 - 추론 과정동안 memory와 계산 시간에 효율적이다.
 - Stochastic gradient descent를 이용해 end-to-end로 트레인 되었으며, 다른 구조에 비해 훈련 가능한 parameter의 수가 훨씬 적다.
 - http://mi.eng.cam.ac.uk/projects/segnet/
 
# Introduction
- 이러한 최근 접근의 일부는 픽셀 단위 라벨링에 category 예측을 위해 제작된 deep architectures에 직접적으로 채택했다.
  - 이러한 결과는 coarse하게 나타났다.
  - 이건 주로 max pooling과 sub-sampling이 feature map resolution을 지웠기 때문이다.
- SegNet은 low resolution feature를 input resolution으로 매핑시킬 필요에 의해서 부상했다.
  - 이러한 매핑이 정확한 boundary localizatio에 효율적인 feature를 생산해야한다.
- SegNet은 pixel-wise semantic segmentation을 위한 효율적인 구조가 되기 위해 구성되었다.
  - model appearance (road, building), shape(cars, pedestrians)에 대한 능력을 필요로 하는 application을 이해하는 road sence에 영향을 받았다.  
  - 그리고 road와 side-walk같은 다른 class들 사이에서 공간-관계(context)를 이해한다. 
  - road같은 large class에 속해있는 pixel의 majority같은 전형적인 road 풍경에서, building과 network는 반드시 smooth segmentation을 해야한다.
  - 이 엔진은 그들의 작은 사이즈에도 불구하고 그들의 형태에 맞춰 object를 delineate할 능력도 갖추어야 한다.
    - 따라서 추출된 이미지 표현에서 **boundary information을 유지하는 것**은 중요해진다.
  - 계산적인 측면에서, network가 효율적이게 되기 위해선 추론을 하는 동안 메모리와 계산하는 시간을 낮추는 것이 필수적이다.
  - 모든 weight를 최적화하기 위해 효율적인 weight update 기술(SGD같은 기술)을 써서 end-to-end로 계산하는 것은 반복을 쉽게 할 수 있기 때문에 추가적인 이득을 볼 수 있다. 
- VGG16과 위상학상으로 동일하다.
  - VGG16에서 fully connected layer를 지운다.
  - SegNet encoder를 더 작게 하고, 더 쉽게 훈련할 수 있도록 한다.
 - SegNet의 핵심 요소 : Decoder network
  - Decoder Network엔 각각의 encoder와 상호응답하는 decoder의 계층으로 구성되어있다.
  - 적절한 decoder는 상호 응답하는 encoder에게서 받은 max-pooling indeces를 사용한다.
  - 이는 input feature map에 대해 non-linear upsampling을 수행하기 위함이다.
  - 이 아이디어는 unsupervised feature learning에서 영향을 받았다.
  - max-pooling indeces를 재사용하는 것에 대한 이점
    1. boundary delineation을 향상시킴
    2. end-to-end training을 가능하게 하는 파라미터의 수를 줄임
    3. 매우 적은 수정으로도 upsampling의 형태가 encoder-decoder 구조로 통합될 수 있음 
- 특징
  - FCN(Fully Convolutional Network)와 SegNet decoding tech에 많은 노력을 기울림
    - segmentation architecture에서 실용적인 trade-off를 전달하고 싶었기 때문
  - 대부분의 deep architecture은 동등한 encoder network를 가지고 있지만, decoder network(training & inference)가 다르다
    - 수백만개의 순서를 가진 trainable parameter를 가지고 있고, 그래서 end-to-end 트레이닝을 하기에 어려움이 있다.
    - 따라서 multi-stage training 방식으로 학습됬다.
    - 대부분의 network는 pre-trained architecture(FCN) 같은 방식으로 추가했다.
    - region proposals for inference 같은 방식으로 supporting aids를 사용한다.
    - classification의 학습과 segmentation network를 분리시킨다.
    - pre-training이나 full training을 위한 additional training data를 사용
    - post-processing 기술을 촉진하는 performance는 유명해지고 있다.
    - 다른 factor가 성능을 향상시킬 수는 있겠지만, 그들의 양적인 결과에서 좋은 성능을 성취하기 위해 반드시 필요한 key design factor를 분리하는 것은 어렵다.
    - 그래서 decoding process를 분석하고, 장점과 단점을 드러내기로 했다.
- 개론
- Sec 2. 최근 논문과 관련된 내용을 리뷰
- Sec 3. SegNet architecture와 그에 대한 분석
- Sec 4. SegNet performance를 평가함
- Sec 5. 미래 연구
- Sec 6. 결론

# Literature Review
<생략>

# Architecture
+ pixelwise classification layer로 가기 전에 encoder network와 corresponding decoder가 있다.
+ Fig3로 이 아키텍쳐가 표현된다.
+ Encoder network 
  - 처음의 object classification을 위해 디자인된 VGG16 network 안에 있는 13개의 convolutional layer에 상호응답하는 13개의 convolutional layer로 구성됨
  - training process를 큰 데이타 셋에서 classification을 위해 학습된 weight로부터 초기화한다.
  - 가장 깊은 encoder output에서 높은 resolution feature map을 유지시키기 위해 fully connected layer를 버릴 수 있습니다.
    - SegNet encoder network의 파라미터 갯수를 줄입니다.
  - 각각의 encoder network는 feature map 집합을 생성하기 위해서 filter bank와 함께 convolution을 수행합니다.
    - batch normalized 되어야 합니다.
  - 각각의 element별로 rectified linear non-linearity (ReLU) max(0,x)가 적용됩니다.
  - max-pooling with a 2x2 window and stride 2 (non-overlapping window)가 수행됨.
    - max-pooling은 input image의 작은 공간 이동을 넘어선 translation invariance를 성취하기 위해서 쓰임.
  - 결과로 나온 output은 2의 배수로 sub-sample됨
    - sub-sampling은 feature map에서 각각의 pixel을 위해 large input image context (spatial window)를 초래함 
  - max-pooling과 sub-sampling의 여러 layer들이 robust classification에 대해 더 translation invariance를 얻을 수 있지만, feature map의 spatial resolution의 loss가 발생한다.
  
- Decoder network
  - 각각의 encoder layer는 corresponding decoder layer를 가지고 있습니다.
  - 따라서 decoder layer는 총 13개 (encoder layer가 13개 이므로)
  - 마지막 decoder output은 각 픽셀에 독립적으로 class probability를 생산하기 위해서 multi-class soft-max classifer에게 먹여진다.

