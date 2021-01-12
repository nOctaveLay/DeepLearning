# Rethinking Atrous Convolution for Semantic Image Segmentation

## DeepLab v3

1. DeepLab v3는 DeepLab v1에서 발전한 모델이므로, DeepLab v1, DeepLab v2에 대한 논문을 봐야 한다.
2. Atrous Convolution 에 대한 이해가 필요하다.
3. [출처](https://arxiv.org/pdf/1706.05587.pdf)

## Abstract

- Revisit atrous convolution
  - Powerful tool to adjust filter’s field-of-view
  - Powerful tool to control the resolution of feature responses computed by Deep Convolutional Neural Networks (DCNN)
- Atrous conbolution을 cascade 하거나 parallel하게 적용시킬 수 있는 module을 디자인 해야 한다.
  - Multiple scale에서 segmenting object[오브젝트 분리]에 대한 문제를 해결하기 위함
  - Multiple atrous rate를 적용함으로써 Multi-scale context를 잡기 위함
- 기존에 제시된 Atrous Spatial Pyramid Pooling module(multiple scale에서의 convolutional feature를 증명하는 것)을 증강시키는 것을 목적으로 해야 한다.
-	Global context를 해독하는 이미지 level feature와 함께 가야 함
-	이는 Performance를 향상시킨다.
4.	Implementation detail을 상세하게 설명
5.	System을 train하는 경험을 공유
6.	DenseCRF라는 선행 processing없이 이전의 DeepLab version을 향상시킴
7.	다른 최신 모델들과 PASCAL VOC 2012 semantic image segmentation benchmark에서 비교할 수 있을만한 성과를 얻음

## Introduction
Deep Convolutional Neural Network를 적용하는데 있어서 두 가지 challenges에 대해 고려해야 한다.
1.	Reduced Feature Resolution
- Consecutive pooling operations나 convolution striding에 의해 일어난다.
- 이는 DCNN이 증가하는 추상적인 특징 표현을 학습하는 것을 허가한다.
- 하지만 이런 local image에 대한 invariance는 세부적인 공간 정보를 필요로 하는 dense prediction 업무를 방해한다. 
- 이런 문제를 극복하기 위해서, atrous convolution을 사용했다.
- 이 Atrous convolution은 semantic image segmentation에 효과적이라고 한다.

**Atorus convolution**
- Dilated convolution이라고 불린다.
- ImageNet의 사용을 변경하는 것을 허용한다.
- ImageNet 
- 마지막 몇 개의 layer의 downsampling operations을 지우고, corresponding filter kernel의 upsampling을 지움으로써denser feature map을 추출하기 위해 미리 학습된 네트워크이다.
- 이는 hole을 filter weights사이에 삽입하는 것과 동등하다.
- Learning extra parameter를 필요로 하는 것 없이Feature responses가 DCNN에서 계산되는 Resolution을 control할 수 있다.

2.	Multiple scale에서 object의 존재를 도출하는 것
- DCNN은 각각의 scale input을 위해 특징을 뽑아내기 위해서 image pyramid에 적용된다.
  - 다른 scale에 있는 object들이 다른 feature map들에서는 현저하게 눈에 띈다.
  - Encoder-decoder 구조는 다양한 scale feature을 decoder part에서 얻습니다.
  - 추가적인 모듈은 긴 범위의 정보를 잡기 위해서 original network의 위로 cascaded된다.
  - 특히, DenseCRF는 pixel level pairwise similarities를 encode하기 위해서 채택되었다.
  - while [59, 90] develop several extra convolutional layers in cascade to gradually capture long range context.
  - 하지만 점층적으로 (cascade) DenseCRF는 점진적으로 긴 범위의 context를 capture하기 위해서 여러 개의 추가적인 convolutional layer를 발전시켰다. 
  - Spatial pyramid pooling은 multiple rate와 multiple effective field of view에서 filter를 가지고 들어오는 feature map과 pooling operation들을 증명한다.
  - 그러므로 다양한 scale에서 object를 capture한다.
- 이 작업에서, atrous convolution을 다시 적용한다.
  - 이는 다양한 크기의 context를 하기 위해서 cascaded module들과 spatial pyramid pooling의 framework 안에서 효과적으로 filter view의 field를 확장시킨다.
- 제시된 모듈은 다양한 비율과 훈련 되야 하는 batch normalization layer를 갖고 있는 atrous convolution으로 되어있다. 
  - Module을 cascade 혹은 parallel로 lay out하는 것으로 실험했다.(특히, Atrous Spatial Pyramid Pooling (ASPP) method) 
  - 3 x 3 atrous convolution을 극단적으로 큰 rate로 적용할 때 중요한 practical issue가 있다.
    - Practival issue : 효과적으로 단순히 1x1 convolution으로 degenerating 할 때 image boundary 효과 때문에 긴 범위의 정보를 잡는 데 실패하는 것. 
    - image-level feature들을 ASPP 모듈로 통합하기 위해 제시
  - 제시된 모델의 훈련에 대한 Implementation detail과 경험 공유
    - 특별하고 잘 주석이 적혀있는 (finely annotated) object들을 다루는 단순히 효과적인 bootstrapping method를 포함
    - 결론적으로, “DeepLabv3”는 이전의 작업을 향상시켰으며, PASCAL VOC 2012에서 85.7%의 performance를 가진다.
 
## Related Work
-	Semantic Segmentation에서 Pixel을 올바르게 classify하는 데에 있어 global features 또는 contextual interaction들은 유용하다.
-	Semantic segmentation을 위한 context information을 얻는 FCNs의 4가지 type에 대해 다뤄볼 것
- Image pyramid
  - 같은 모델, 일반적으로 weight를 공유함
  - Multi-scale input에 적용됨.
  - 작은 크기의 input에서 feature response가 긴 범위의 context를 암호화함.
  - 큰 크기의 input은 작은 object detail을 보존
  - 전형적인 예는 input image를 Laplacian pyramid를 통해 변환시키고, 각각의 scale input을 DCNN에게 주고, 모든 크기의 feature map을 합치는 Farabet et al.을 포함한다.
    - [19,69]는 연속적으로 coarse-to-fine으로부터 multi-scale input을 적용한다.
    - [55,12,11]은 직접적으로 input을 다양한 scale에 대해 resize 한다.
    - 이런 타입의 모델의 단점은 더 크고 더 깊은 DCNN들에 대해 이미지의 크기를 변형시키지 못한다는 것이다. (networks like [32,91,86]
    - GPU 메모리의 한계 때문, 그리고 inference stage 동안 이게 일반적으로 적용된다.
- Encoder-decoder
  - 이 모델은 encoder와 decoder로 이루어진다.
  - **Encoder**
    - Feature map의 공간적인 차원이 점진적으로 감소
    - 더 긴 범위의 정보가 더 깊은 encoder output에서 더 쉽게 잡아짐.
  - **Decoder**
    - Object detail과 공간 차원이 점진적으로 회복됨
    - [60,64]는 low resolution feature의 upsampling을 배우기 위해서 deconvolution을 적용
  - SegNet [3]은 pooling indices를 encoder에서 재사용하고, feature response를 densify 하기 위해서 추가적인 convolutional layer를 배움.
  - U-Net은 encoder feature에서 corresponding decoder activation으로 skip connections을 더한다. 그리고 Laplacian pyramid reconstruction network를 사용한다.
  - 더 최근에, RefineNet [54] 와 [70,68,39]는 model의 effectiveness를 several semantic segmentation benchmark에 대한 encoder-decoder 구조에 바탕을 두고 설명해왔다. 
    - 이런 타입의 모델은 object detection의 문맥에서도 감지된다.
- Context module
  - 이 모델은 긴 범위의 context를 암호화 하기 위해서 계단식으로 보여지는 추가적인 모듈을 포함한다.
  - 효율적인 방법 중 하나는 높은 차원의 filtering algorithm을 갖고 있는 DenseCRF를 DCNN으로 통합하는 것입니다.
  - 더욱이 [96,55,73]은 여러 개의 부가적인 convolutional layer들을 DCNN의 belief maps들의 위에 올리는 것 대신에 CRF와 DCNN 요소를 jointly하게 train하는 것을 제시합니다.
  - 반면에 [59,90]은 몇 개의 다른 convolutional layer를 DCNN의 belief map의 위에 올립니다.
    - Belief map들은 예측된 class들의 수와 같은 수의 output channel들을 포함하는 마지막 DCNN feature map들입니다.
    - Context information을 capture하기 위함입니다.
- 최근에, [41]은 일반적이고 공간적으로 높은 차원을 가지고 있는 convolution을 배우는 것을 목표로 했습니다.
- 그리고 [82,8]은 semantic segmentation을 위해 Gaussian Conditional Random Fields를 DCNN들과 결합했습니다.
- **Spatial pyramid pooling**
  - 이 모델은 여러 범위에서 context를 잡기 위해서 spatial pyramid pooling을 사용합니다.
  - Image-level 특징들은 global context information을 위해 ParseNet에서 얻어집니다.
  - DeepLabv2[11]은 atrous spatial pyramid pooling (ASPP)를 제시했습니다.
    - 다른 비율들을 가진 Parallel atrous convolution layer들은 ASPP에서 다양한 크기의 정보를 얻었습니다.
    - 최근에, Pyramid Scene Parsing Net (PSP)는 여러 개의 grid scale에서 공간적인 pooling을 수행하고, 다양한 semantic segmentation benchmark에서 outstanding performance를 설명한다.
    - Global context를 묶기 위해서 LSTM에 기반을 둔 다른 method들이 있다.
  - Spatial pyramid pooling은 object detection에서 또한 적용된다.
    - atrous convolution을 spatial pyramid pooling을 위한 툴과 context module로서 주로 탐방한다.
- Deeplabv3는 일반적으로 어떤 network에도 적용될 수 있다.
  - ResNet의 original 마지막 block의 여러 개의 copy들을 복제했고, 이를 계단식으로 배치했으며, ASPP 모듈을 재 방문했다.
  - ASPP 모듈은 여러 개의 atrous convolution을 병행으로 한다.
  - 우리의 계단식 모듈은 belief map 대신에 직접적으로 feature map에 관여하는 것을 기억해라.
  - 실험적으로 batch normalization을 훈련하는 가장 중요한 방법을 찾았다.
  - Global context를 더 잡기 위해서, augment ASPP를 image-level feature에 제시했다. [58,95]와 유사한 방식이다.
- Atrous convolution:
  - Atrous convolution에 기반한 model들은 active하게 semantic segmentation을 위해 쓰여진다.
  - 넓은 범위의 정보를 잡기 위해서 Atrous rate를 수정한 효과에 대한 실험들은 ResNet의 마지막 두 개의 block에 hybrid atrous rate를 적용했다.
  - 반면에 더욱더 학습된 offset을 가지고 input feature을 sample하는 deformable convolution을 배우도록 제시했다. 이는 atrous convolution을 일반화한다.
  - 더 segmentation model 정확성을 향상시키기 위해서, 
    - [83]은 image caption을 얻었다. 
    - [40]은 video motion을 활용했다.
    - [44]는 depth정보를 통합한다.
    - Atrous convolution은 [66,17,37]에 의해 object detection이 적용된다.

## Methods
-	Semantic segmentation에서 어떻게 atrous convolution이 dense feature들을 추출 하는 데에 적용되는지를 review
-	제시된 모듈을 계단식 혹은 병행으로 실행된 atrous convolution module로 토론할 것이다.
### Atrous Convolution for Dense Feature Extraction
  - Fully convolutional fashion에서 deploy된 DCNNs은 semantic segmentation의 일에서 효과적으로 보여진다.
  - 하지만, max-pooling과 연속적인 layer에서 striding의 반복된 combination은 feature map들의 spatial resolution을 줄인다.
  - 이는 최근 DCNN [47,78,32] 에서 각각의 방향을 거친 32개의 factor에 의한다.
  - Deconvolutional layer(또는 transposed convolution) [92,60,64,4,71,68]은 공간적인 해상도를 회복시키기 위해서 employ된다.
  - 대신에, 우리는 atrous convolution의 사용을 주장한다.
  - Atrous convolution은 undecimated wavelet transform의 효과적인 계산을 위해 개발되었다.
    - [26,74,66]에 의하면 이전에 DCNN context 안에서 사용되었다. 
    - 2차원 시그널에서 각각의 location을 i, output을 y, 그리고 filter를 w 라고 하자.
    - Atrous convolution은 input feature map x에 대해 적용된다.
    - Atrous rate, r이 input signal을 sample한 것을 갖고 있는 stride에 상호 반응해야한다.
    - 이는 input x를 2개의 연속된 filter value들을 각각의 공간 차원에 따라서 r-1개의 zero를 넣음으로써 생산된 upsampled filter와 convolve 연산을 하는 것과 같다.
    - (따라서 atrous convolution의 이름은 영어에서 hole을 의미하는 프랑스 언어 trous에서 왔다.)
    - 표준 convolution은 rate r = 1인 special한 case이다.
    - atrous convolution은 rate value를 수정함으로써 적합하게 filter의 field-of-view를 수정하도록 했다.
    - Fig 1 for illustration
    - [Fig 1]
    - Atrous convolution은 fully convolutional network에서 고밀도로 feature response를 계산하는 방법을 명확히 제어할 수 있도록 해준다. 
  - output_stride로 입력 이미지의 공간 resolution을 최종 출력 resolution의 비율로 나타낸다.
  - 최종 feature response는 (fully connected layer 또는 global pooling 전에) input image dimension보다 32배 작다. 
    - 이는 DCNN [47,78,32]은image classification을 위해 deploy 됐기 때문이다.
    - 그러므로 output_stride = 32
  - 만약 DCNN에 있는 계산된 feature response들의 공간적 밀도를 2배로 하기를 원한다면, 해상도를 감소시키는 마지막 pooling 또는 convolutional layer의 stride는 1로 설정되어야 한다.
    - 이는 signal decimation을 피하기 위함이다.
    - 그 다음, 모든 뒤에 따라오는 convolutional layer들은 atrous convolutional layer로 대체되어야 한다.
  - Rate = 2
    - 이건 추가적인 parameter로 학습하는 것 없이 더 밀도 있는 feature response를 추출하기 위함이다.
    - 더 많은 detail을 원한다면 [11]을 참조
    
### Going Deeper with Atrous Convolution
  - 계단식으로 보여지는 atrous convolution을 가지고 module을 designing한다.
  - 더 자세하게 말하면, 마지막 ResNet block의 여러 개의 copy를 복사한다.
  - 이는 Fig 3에 기재되어있다.
  - [Fig 3]
  - 그리고 이들을 계단식으로 재배치 합니다.
  - 3x3 convolution들이 이들의 block에 있습니다.
  - 마지막 convolution은 마지막 block에 있는 하나를 제외하고 stride 2를 포함합니다. 이는 original ResNet과 같습니다.
  - 이 소개된 striding이 더 깊은 block들에서 긴 범위의 정보를 잡기가 쉬웠습니다.
    - 예를 들어, Fig3 (a) 에 나온 것처럼 전체 이미지 feature은 마지막 작은 resolution feature map으로 요약될 수 있습니다.
  - 하지만 우리는 연속적인 striding이 semantic segmentation에서 매우 좋지 않다는 것을 알 수 있습니다. 
    - 왜냐하면 세부 정보들이 decimate되기 때문입니다.
  - 그래서 atrous convolution을 요구되는 output_stride value에 의해 결정된 rate로 적용시켰습니다. 
  - 이 때의 output_stride = 16이며, 이는 Fig 3(b)에서 볼 수 있습니다. 
  - 이 제시된 모델 안에서, 우리는 cascaded ResNet block들을 7까지 실험합니다. 
    - 이 때의 추가적인 block5, block6, block7은 block4의 복제품입니다.
    - 이 때의 Output_stride = 256입니다.
    - 이 때 어떤 atrous convolution도 적용되지 않았다고 가정합니다.

### Multi-grid Method
  - 다른 사이즈의 grid에 계층을 적용한 multi-grid method에 동기를 얻음
  - Block4 ~ block7안에 제시된 모델 안에서 다른 atrous rate를 적용
  - 특히 block4 ~ block4안에 있는 convolutional layer들에게 unit rate들을 Multi_Grid = (r1,r2,r3)로 정함.
  - Convolutional layer에서 마지막 atrous rate는 unit rate와 그에 따르는 (corresponding) rate의 곱과 동일하다.
    - 예를 들어, output_stride = 16이고, Multi_Grid = (1,2,4)일 때, 3개의 convolution은 rates = 2 • (1, 2, 4) = (2, 4, 8) 를 block4에 각각 가진다.
  - Atrous Spatial Pyramid Pooling
    - [11]에서 제시된 Atrous Spatial Pyramid Pooling을 재 방문한다.
    - [11]에서는 다른 atrous rate들을 가진 4개의 parallel atrous convolution이 feature map의 위에 적용된다.
    - ASPP는 spatial pyramid pooling의 성공에 영향을 받았다.
      - Spatial pyramid pooling은 다른 scale에서feature을 resample하는 것이 효과적이라는 것을 보여준다.
      - 이는 자의로 정한 scale에서의 범위를 효과적이고 정확하게 classify하기 위함이다.
      - [11]과는 다르게, ASPP에 batch normalization을 포함시킨다.
    - 다른 atrous rate를 가진 ASPP는 multi-scale information을 잡는다.
      - 하지만, sampling rate가 커지면 커질수록, valid filter weight의 수는 점점 작아집니다.
      - Valid filter weight = zero로 패딩된 것 대신 valid feature region에 적용되는 weight들
    - 이 효과는 3x3 filter를 65x65 feature map에 다른 atrous rate로 적용시켰을 때 Fig.4에서 나타난다.
    - [Fig 4]
    - Rate value가 feature map size와 가까운 극단적인 상황에서, 3x3 filter는 전체 image context를 잡는 대신에 1x1 filter로 degenerate된다.
      - 이는 오직 center filter weight가 효과적일 때에만 가능하다.
    - [58,95]와 비슷한 image level feature를 적용시킨다.
      - 위에서 언급한 문제를 극복하기 위함이다.
      - model에게 global context 정보를 통합하기 위함이다.
    - 특히, global average pooling을 모델의 마지막 feature map에 적용시킬것이다.
    - 그 후 결과 image-level 특징들을 1x1 convolution에 256개의 filter로 줄 것이다.
    - 그 다음 bilinear하게 feature을 요구된 공간 차원으로 upsample할 것이다.
    - 마지막에, 향상된 ASPP는 output_stride = 16일 때 1개의 1x1 convolution과 3개의 3x3 convolution으로 rate = (6,12,18)로 구성한다.
      1.	256 filter와 batch normalization이 위의 과정에서 무조건 들어간다.
      2.	Image-level feature는 Fig 5.에서 보여진다.
      3.	Output_stride = 8일 때 그 rate들이 double이 된다.
      4.	모든 branch들로부터 나온 결과적인 feature은 합쳐집니다.
      5.	그리고 final logit을 생산하기 전의 마지막 1x1 convolution을 지나가기 전에 또 다른 1x1 convolution을 지나갑니다. (당연하지만 256 filter와 batch normalization)

## Experimental Evaluation
-	미리 학습된 ImageNet과 ResNet을 atrous convolution을 적용시킴으로써 semantic segmentation에 맞게 변형시켰다.
  -	이는 더 정밀한 feature들을 뽑아내기 위함이다.
-	Ouput_stride가 최종 output resolution에 대한 input image spatial resolution의 비율로 정의된다라는 점을 명심해라.
-	예를 들어, output_stride = 8일 때, 마지막 두 개의 block은 (우리의 notation에서 block 3와 block 4) original ResNet에서 rate = 2와 rate = 4인 atrous convolution을 각각 포함한다.
-	Tensorflow로 구현되었다.
-	PASCAL VOC 2012에서 semantic segmentation benchmark에서 제시된 모델을 평가한다.
  - PASCAL VOC 2012는 20개의 foreground object class와 1개의 background class로 구성되어 있다.
  -	Original dataset은 1,464(train) 1,449(val) 그리고 1,456(test)의 training, validation, testing을 위한 pixel-level labeled image로 구성되어 있다.
  -	Dataset은 [29]에 의해 제공된 추가적인 annotations으로 확장됩니다.
  -	이 dataset은 10,582개의 training image(trainaug)를 생산합니다.
  -	Performance는 21개의 class를 통해 평균이 내진 pixel IOU(intersection-over-union)으로 측정됩니다.

1.	Training Protocol
  - Learning rate policy
    - [58,11]과 유사하게, 우리는 “poly” learning rate policy를 employ 해야 한다.
    - 처음의 learning rate는 (1 − iter /max iter ) ^ (power) with power = 0.9.
  - Crop Size
    - 일반적인 training protocol[11]을 따르면, patch들은 training 중 image에서 모아진다.
    - 큰 rate를 가진 atrous convolution이 효과적이기 때문에, 크게 자른 size가 무조건 필요해진다.
    - 다른 말로 하자면, 큰 atrous rate를 가진 filter weight가 대부분 pad된 zero 영역에 적용된다.
    - 그러므로 training 과 test를 할 동안 crop size를 513으로 적용시켰다.
  - Batch normalization
    - ResNet의 위에 있는 추가된 모듈은 모두 batch normalization parameter를 갖고 있다.
    - 이는 잘 훈련되기 위해서 매우 중요한 사항이었다.
    - 큰 batch size는 batch normalization parameter를 훈련하기 위해서 필요로 되어지기 때문에 output_stride = 16으로 뒀고, batch normalization에서 batch size 16으로 계산한다.
    - Decay = 0.9997을 가지고 batch normalization parameter를 학습시킨다.
  - 첫 번째 훈련
    - Training set으로 30k의 iteration과 learning rate = 0.007로 훈련을 한 다음에,
    - batch normalization parameter을 고정시켰다.
    - Output_stride = 8을 employ했다.
  - 공식적인 PASCAL VOC 2012 trainval set에서 또 다른 30K iteration으로 학습시켰다.
    - 그 때는 더 작은 base learning rate = 0.001을 가졌다.
    - Atrous convolution은 output_stride를 다른 훈련 과정에서 다른 model parameter를 학습하는 것 없이 control할 수 있게 허락해준다.
    - Output_stride = 16이 output_stride = 8보다 빠르다.
      -	Intermediate feature map이 공간적으로 4배나 줄기 때문이다.
      - 하지만 정확도의 희생은 더 거친 feature map을 준다.
   - Upsampling logits
    - [10,11]이라는 과거의 작업에 의하면, target의 ground truth는 8로 downsample된다. 
      - 이는 훈련 도중 진행된다.
      - Output_stride = 8이다.
    - Final logit을 upsample하는 것보다 Groundtruth를 손상되지 않게 유지하는 것이 중요하다.
      - Groundtruth를 downsampling 하는 것이 올바른 annotaion을 제거하기 때문이다.
      - 이는 세부적인 부분의 back-propagation이 불가능하게 만든다.
  - Data augmentation
    - Input image를 0.5배에서 2배 사이에서 random하게 크기를 조절했다.
    - 훈련 도중 왼쪽과 오른쪽을 랜덤하게 뒤집었다.
2.	Going Deeper with Atrous Convolution
- ResNet-50
  - Tab.1에서, ResNet-50에 적용시킬 때의 output_stride의 효과를 실험했다.
  - Block7이 존재한다. (다시 말하자면, 추가적인 block5, block6, block7존재)
  - table에서 보여주다시피, output_stride = 256일 경우 다른 것들보다 performance가 훨씬 더 악화된다.
    - 이는 atrous convolution이 적용되지 않았을 때다
    - 이는 Severe signal decimation 때문에 더 나빠진다.
  - Output_stride가 더 커질때, 그리고 correspondingly하게 atrous convolution을 적용할 때, performance가 20.29%에서 75.18%로 증가했다. 
    - 이는 semantic segmentation에서 cascadedly하게 block을 더 많이 쌓을 때 atrous convolution이 필수적임을 보여준다.
- ResNet-50 vs ResNet-101
  - ResNet-50을 더 깊은 network ResNet-101로 대체한다.
  - Cascaded block의 수를 바꾼다.
  - Tab.2에서 보여준 것처럼, 더 많은 block이 추가될수록 성능이 증가한다.
  - 그렇지만, 향상의 이득이 점점 줄어든다.
  - ResNet-50에서 block7을 employ 하는 것은 performance를 살짝 감소시킨다.
  -  하지만 여전히 ResNet-101의 performance보다 높다.
- Multi-grid
  - Multi-grid method를 Tab.3안의 여러 개의 block이 cascaded하게 더해진  ResNet-101에 적용시킨다. 
  - Multi_Grid = (r1,r2,r3) = The unit rates
  - Unit rate는 block 4 그리고 이 이후에 추가된 block들에게 모두 적용된다. 
  - table에서 보여지듯, multi-grid method가 된 (a)가 일반적인 version보다 일반적으로 더 좋다.
    - 일반적인 version : (1,1,1)을 적용시켰을 경우.
    - (2,2,2)는 별로 효과적이지 않았다.
    - 가장 좋은 모델은 (1,2,1)이 적용되었을 때였다.
- Interference strategy on val set
  - 제시된 모델은 output_stride = 16으로 훈련된 것이었다.
  - 추론 과정 동안, 더 정밀한 feature map을 얻기 위하여 output_stride = 8로 적용시켰다.
  - Tab. 4에서 보여지듯, 흥미롭게도, 가장 cascaded model을 평가했을 때 output_stride를 16으로 했을 때보다 1.39%의 성능이 나아졌다.
  - 성능은 다양한 크기의 input(0.5, 0.75, 1.0, 1.25, 1.5, 1.75)과 좌우 반전 이미지에 추론을 수행하기 위해 향상됩니다.
  - 특히, 마지막 결과로서 평균적인 확률을 각각의 크기와 뒤집힌 이미지로부터 구한다.
3. Atrous Spatial Pyramid Pooling
  - [11]과는 주요한 차이점을 가진 Atrous Spatial Pyramid Pooling (ASPP) 모듈을 가지고 실험한다.
    - 이는 batch normalization parameter가 fine-tuned 되어 있고, image-level feature들은 포함되어있다.
  - ASPP
    - 향상된 ASPP module에서 우리는 Tab.5에서 봐서 알 수 있듯 block4에 있는 multi-grid를 통합하는 것에 대한 효과를 실험한다. 
    - 첫 번째로, ASPP = (6,12,18)로 고정했다.
      - 즉, rates = (6,12,18)이라는 뜻이다.
      - Tab 3에 있는 block4 column과 비교해라
    - 만약 더 넓은 범위의 context를 위해 rate가 24인 다른 parallel branch를 추가적으로 employ한다면, 성능은 0.12%만큼 하락한다.
    - 반대로, image-level feature을 가진 ASPP 모듈을 증강하는 것은 효과적이었다.
      - 77.21%의 결과 성능을 보여줬다.
- Inference strategy on val set
  - 유사하게, 우리는 output_stride = 8을 일단 모델이 훈련될 때 추론 과정에서 적용시킨다.
  - Tab.6 에서 보이다시피, output_stride = 8은 output_stride = 16을 썼을 때보다 1.3%의 향상을 가져온다.
    - 다양한 크기 input에 적용된다.
    - 좌우 반전된 이미지를 더하는 것은 각각 0.94%와 0.32%로 성능을 더욱더 향상시킨다.
    - ASPP의 가장 좋은 모델은 79.77%의 성능을 가지고 있다.
      - 이는 cascaded atrous convolution module들보다 더 좋다. 
      - 이는 test set evaluation에서 우리의 마지막 모델로 선정되었다.
- DeepLabv2와 비교했을 때
  - 우리의 가장 cascaded가 잘 된 모델 (in Tab. 4) 그리고 ASPP 모델 (in Tab. 6)는 PASCAL VOC 2012 val set에서 이미 DeepLabv2를 능가했다.
    - DenseCRP post-processing또는 MS-COCO pre-training이 없는 두 가지의 경우에서 측정
    - DeepLabv2는 DenseCRF와 MS-COCO를 썼음에도 77.69%가 나왔다.
  - 제시된 모델 안에서 batch normalization parameter들을 fine-tuning하고 포함하는 것이 향상에 영향을 많이 끼쳤다.
  - 그리고 또한 많은 크기의 context를 암호화하는 더 좋은 방법을 갖는 것도 향상에 많은 영향을 끼쳤다.
4.	Appendix
- [14]에 대한 결과에서 Cityscape와 hyper parameter의 효과 같은 실험적인 결과
5.  Qualitative results
- Fig. 6.에서 우리의 가장 좋은 ASPP의 양적인 visual result를 줄 것이다.
- figure에서 보여지듯, 우리의 모델은 어떤 DenseCRF 없이 object를 매우 잘 분리하는 것이다.
6.	Failure mode
  - Fig. 5의 밑의 줄에서 보여지듯이, 우리의 모델은 sofa와 chair, dining table and chair, and rare view of objects를 구분하는데 어려움을 겪는다.
7.	Pretrained on COCO
- 다른 최신의 모델과 비교할 때, 우리는 가장 좋은 ASPP model을 MS-COCO dataset으로 미리 학습한다.[57]
  - MS-COCO train-val_minus_minival set에서부터, 우리는 1000개의 픽셀 이상을 갖고 있는 annotation region들을 갖는 image만 고를 것이다.
  - 그리고 PASCAL VOC 2012에서 정의된 class들을 포함할 것이다.
  - 훈련에 약 60K개의 이미지를 초래한다.
- 게다가, PASCAL VOC 2012에서 정의되지 않은 MS-COCO class들은 모두 background class로 분류할 것이다.
- MS-COCO dataset을 미리 훈련한 후에, 제안된 모델은 val set에서 82.7%의 성능을 보여줬다.
  - 이 때의 output_stride = 8이며, multi-scale input을 사용한다.
  - 추론 중에 좌우 반전된 이미지를 더한다.
  - 초기 learning rate = 0.0001을 적용, 같은 training protocol을 사용.
    - 이는 PASCAL VOC 2012 dataset에서 fine-tuning이 되었을 때
8.  Test set result and an effective bootstrapping method
- PASCAL VOC 2012 dataset은 증강된 dataset보다 더 높은 quality의 annotation을 제공한다.
- 우리의 모델을 PASCAL VOC 2012 trainval set에서 더 fine-tune했다.
- 특히, 우리의 모델은 output_stride = 8로 훈련되었다.
  - 그래서 annotation detail이 유지되었다.
- 게다가, pixel hard example mining을 [85,70]처럼 하는 것 대신에, 우리는 hard image에 bootstrapping을 했다.
- 특히, 우리는 training set에서 hard class를 담고 있는 (특히 자전거, 의자, 테이블, potted plant, 그리고 소파 같은) image를 duplicate한다.
- Fig.7에서 보여지듯, 가장 단순한 bootstrapping method는 bicycle class를 segmenting하는데에 유용하다.
- 마지막으로, DeepLabv3는 test set에서 Tab.7에서 보여지듯 어떤 DenseCRF post-processing없이 85.7%의 성능을 성취했다.
9.	Model pretrained on JFT-300M:
- [79]에 대한 최신 연구에 의해 동기 부여 받아서, 우리는 ImageNet과 JFT-300M에 미리 훈련된 ResNet-101 모델을 고용했다.
- 이는 PASCAL VOC 2012 test set에서 86.9%의 성능을 보였다.

## 결론
-	Dense feature map을 추출하고 큰 범위의 context를 잡기 위해서 “DeepLabv3” upsampling filter와 같이 있는atrous convolution을 employ한다.
-	특히, 다중 스케일 information을 encode하기 위해서, 우리의 제시된 cascaded module은 점진적으로 atrous rate를 2배로 합니다.
-	반면에 우리의 제시된 image-level feature가 증강된 atrous spatial pyramid pooling module은 다양한 sample rate에서의 filter와 효과적인 field-of-view들을 가진 feature을 증명합니다.
-	우리의 실험적인 결과는 제시된 모델이 이전의 DeepLab version을 매우 향상시킨다는 것과 다른 최신의 PASCAL VOC 2012 semantic image segmentation benchmark에 비견할만한 성장을 얻는다는 것을 보여줍니다.
1.	Effect of hyper-parameters
A.	Main paper와 같은 training protocol을 따른다.
B.	몇 개의 hyper-parameter의 효과를 실험한다.
C.	New training protocol
i.	[10,11]에 있는 training protocol을 바꿨다.

 
조사해야 되는 논문 : RefineNet, ParseNet
