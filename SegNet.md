
# SegNet Review
- SegNet : A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
- https://arxiv.org/pdf/1511.00561.pdf

# 0. Abstract
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
 
# 1. Introduction
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

# 2. Literature Review
<생략>

# 3. Architecture
+ pixelwise classification layer로 가기 전에 encoder network와 corresponding decoder가 있다.
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
  - 만약 추론을 하는 동안 메모리가 제한되어 있지 않으면, 모든 encoder feature map은 sub-sampling 후에 저장된다.
    - practical application한 경우가 아니며, 이러한 정보를 저장할 훨씬 더 효율적인 방법을 제시해야 한다.
      - max-pooling *indice*만을 저장하는 것을 포함
    - 즉, 각각의 pooling window에서 나오는 최대 feature value의 위치는 각각의 encoder feature map에 저장되는 것이다.
      - 실제로, 각각의 2x2 pooling window에 2비트만 사용해서 이것이 가능하다.
    - float precision 안에 feature map을 저장하는 것과 비교하면 훨씬 더 효율적이다.
    - 메모리를 덜 쓰는 방식은 accuracy를 살짝 줄이는 효과를 가져오지만, 현실적으로 적용하는데에는 알맞다.
+ Decoder network
  - 각각의 encoder layer는 corresponding decoder layer를 가지고 있습니다.
  - 따라서 decoder layer는 총 13개 (encoder layer가 13개 이므로)
  - 마지막 decoder output은 각 픽셀에 독립적으로 class probability를 생산하기 위해서 multi-class soft-max classifer에게 먹여진다.
  - 적절한 decoder는 상호응답하는 encoder feature map으로부터 저장된 max-pooling indices를 이용해서 input feature map을 upsample함
    - sparse feature map 생성
  - Fig.3.에서 decoding tech 표현
  - 이러한 feature map은 dense feature map을 생산하기 위해서 train 가능한 decoder filter bank와 convolve 연산을 함
  - batch normalization step은 이러한 맵들 각각에 적용됨.
  - 첫번째 encoder와 상호응답하는 decoder는 multi-channel feature map을 생성 
    - encoder input과 같은 수의 size와 channel을 생성하는 다른 decoder와는 다름.
  - 최종 디코더의 출력에서 고차원 feature 표현은 훈련 가능한 soft-max classifier에 공급된다.
    - 이 soft-max는 각각의 pixel을 독립적으로 classify 한다.
    - soft-max의 output은 K channel개의 확률 이미지이다. (K는 class의 수를 의미한다.)
  - 예측된 segmentation은 각각의 픽셀의 최대 확률을 가진 class와 상호응답한다.
+ loss
  - max-pooling과 sub-sampling의 여러 layer들이 robust classification에 대해 더 translation invariance를 얻을 수 있지만, feature map의 spatial resolution의 loss가 발생한다.  
  - 이미지 표현의 증가하는 lossy는 (boundary detail) boundary delineation이 필수적인 segmentation에 유용하지는 않다.
  - sub-sampling이 수행되기 전에 encoder feature map에 있는 boundary information을 capture하고 저장하는 것은 필수적이다.
+ 유사한 넷
  - DeconvNet과 U-net
  - DeconvNet
    - 훨씬 더 많은 parameterization을 가진다.
    - 더 계산적 자원이 많이 필요하다.
    - convolutional 방법을 택하고 있다 할지라도, end-to-end 학습이 힘들다. (fully connected layer를 사용하고 있기 때문)
    - Sec. 4에서 자세히 비교
  - U-Net (의학적인 image community에 의해 제한됨)
    - pooling indice를 재사용 하지 않는다.
    - 대신, 전체적인 feature map을 corresponding decoder로 옮긴다. (더 많은 메모리 사용 필요)
    - upsampled decoder feature map으로 feature map을 concatenate한다.(합친다)
    - VGG net architecture처럼 conv5나 max-pool 5같은 게 없다.
  - SegNet은 VGG net에서 사용된 convolutional layer weight를 미리 학습된 weight로서 사용한다.

## 3.1 Decoder Variants
- encoder는 다 똑같기 때문에([2][3][4]) decoder만 분석하면 됨
- SegNet-Basic (SegNet의 작은 버전, 4개의 encoder와 4개의 decoder로 구성)을 FCN(decoder variants)과 비교해서 설명하겠슴.
  - 모든 SegNet-Basic에 있는 encoder는 max-pooling과 sub-sampling을 수행
  - corresponding decoder는 받은 max-pooling indices로 input을 upsampling함
  - encoder와 decoder 안에 있는 각각의 convolutional layer 후에 batch normalization이 수행
  - convolution 후에 어떤 bias도 첨부 안됨
  - decoder network에서 ReLU함수도 제시 안됨
  - 7x7 constant kernel size -> smooth labelling을 위한 wide context를 주기 위함
    - 즉, 가장 깊은 layer feature map안의 pixel이 106x106 pixel의 input image 안에 있는 context window로 trace back 할 수 있도록 도와줌
  - 수많은 다른 variant(decoder)를 탐색하도록 도와주고, 적절한 시간에 train할 수 있도록 도와줌
- FCN-basic (FCN의 comparable version)
  - SegNet-Basic과 같은 encoder를 가지고 있음
  - 모든 decoder에 FCN decoding tech를 가지고 있음
- Fig 3 의 왼쪽은 SegNet에서 사용되는 decoding tech이다. (upsampling step에서 어떤 학습도 없을 경우)
  - upsampled map은 공간적 input의 밀도를 높이기 위해서 학습가능한 multi-channel decoder filter와 convolve 연산 됨.
  - 각각의 decoder filter는 upsampled feature map의 수와 동등한 채널의 수를 가지고 있음
  - 더 작은 variant는 decoder filter가 single channel이라는 것이다.
    - 다시 말하자면, 오직 decoder의 corresponding upsampled feature map만 convolve 한다는 것이다.
    - 이런 variant(SegNet-Basic-SingleChannelDecoder)은 훈련가능한 파라미터의 수와 추론 시간을 제거합니다.
- Fig 3.의 오른쪽은 FCN의 decoding technique입니다.
  - FCN 모델의 가장 중요한 디자인 요소는 encoder feature map의 dimensionality reduction step입니다.
  - 이것은 corresponding(상호응답하는) decoder들 안에서 쓰이는 encoder feature map을 *압축*합니다.
  - Dimensionality reduction은, 64개의 채널 중에서, 1 x 1 x 64 x K 의 훈련 가능한 필터와 encoder feature map을 convolving 함으로서 수행됩니다.
  - K : class의 갯수입니다.
  - 압축된 K개의 channel을 가지는 최종 encoder layer의 feature map은 decoder network의 input이 됩니다.
  - 이 network의 decoder안에서 고정되고 훈련가능한 multi-channel upsampling kernel을 사용해서 upsampling이 inverse convolution에 의해 수행된다. (이를 deconvolution이라고 이름붙인다.)
  - **Segnet에서 trainable decoder filter를 사용한 multi-channel convolution은 feature map을 집적(고밀도)화하는 upsampling 후에 수행된다.**
  - 8x8 kernel size 
  - FCN의 upsampled feature map은 K 채널을 가진다.
  - upsampled feature map은 decoder feature map의 output을 생산하기 위하여 그 다음 element별로 corresponding resolution encoder feature map에 더한다.
  - upsampling kernel은 bilinear interpolation weights를 사용해서 초기화된다.
  - FCN decoder model은 추론하는 과정 동안 encoder feature map을 저장하는 것이 필요하다.
  - 이건 embedded application을 위해 memory intensive하다
    - 예를 들어, 180 x 240 resolution에서 FCN-Basic의 첫번째 layer의 64개 feature map을 저장하는 것이 32bit floating point precision에서 11MB를 차지하는 것이다.
    - 이건 11개의 feature map에 대한 dimensionality reduction을 사용하면서 더 작게 만들 수 있다.
  - pooling indices에 대해서 negligible storage의 비용이 필요하다.
    - 만약 2bit를 사용해 2x2 pooling window를 저장한다면, .17MB 필요)
  - endoer feature map의 추가적인 스텝을 버리고 오직 upsampling kernel만 배우는 FCN-basic model의 variant를 생성(**FCN-Basic-NoAddition**)
  - upsampling을 위해 학습할 필요가 없는 고정된 bilinear interpolation weight를 사용해서 upsampling 학습(Bilinear-Interpolation)
  - 다른 끝단에서, 각각의 layer의 64개의 feature map을 SegNet decoder에서 나온 상호 응답하는 output feature map에 더했다. (SegNet-Basic-EncoderAddition)
    - SegNet의 더 메모리에 intensive한 variant를 생성하기 위해서이다.
  - pooling indice들을 upsampling을 위해 사용했으며, 이는 convolution step이 뒤를 잇는다. 
## 3.2 Training

## 3.3 Analysis
