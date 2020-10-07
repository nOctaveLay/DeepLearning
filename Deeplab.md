# Abstract
- Semantic Image segmentation에 관련된 논문
- 실용적으로 쓸 수 있는 실험적인 3개의 기여

  1. **atrous convolution**
      - upsample 된 filter에 convolution을 강조 => dense prediction에 강력한 tool
      - DCNN 안에서 feature response가 계산될 때 resolution(해상도)를 명확히 control하게 만든다.
        => 파라미터의 갯수나 계산의 양을 증가시키는 것 없이 더 큰 context을 통합하기 위해서 filter의 view의 영역을 확장시킬 수 있게 만든다.
  
  2. **ASPP**
      - atrous spatial pyramid pooling (ASPP)
      - 다양한 scale에서 object를 견고하게 분리
      - 다양한 sampling rate와 효과적인 fields-of-views에서 filter를 가진 convolutional feature layer를 증명
        - 따라서 다양한 크기에서 image context 뿐만 아니라 object도 잡음 

  3. **DCNN과 확률적인 그래픽 모델의 method를 결합시킴으로서 object boundary의 localization 향상**
      - 일반적으로 max-pooling과 downsampling의 deploy된 결합은 invariance를 향상, 그러나 localization accuracy를 희생
      - 이를 최종 DCNN layer의 responce와 fully connected Conditional Random Field (CRF)와 결합함으로서 극복
        - 질적, 양적으로 localization performance를 향상
        
# Introduction
- DCNNs에선 invariance 중요
  - 추상적 데이터 표현을 배워야 하기 때문

- 그렇지만 semantic segmentation같이 dense prediction task에서는 필요가 없다.
  - 공간 정보의 추상화가 필요 없기 때문

- 이 논문에선 semantic image segmentation에 있어 DCNN의 3가지 문제점을 고려
  1. 감소된 feature resolution
  2. 다양한 크기에서 object의 존재
  3. DCNN invariance 때문에 localization accuracy가 감소

- 3가지 문제점의 극복
  1. 감소된 feature resoltuion => Atrous convolution으로 극복
    - 원인 
      - 이미지 분류를 위해 처음에 고안된 DCNN의 연속적인 layer에서 실행된 반복된 max-pooling과 downsampling('striding')의 결합으로 인해 발생.
    - 해결
      - DCNN에서 몇 개의 max pooling layer로부터 온 downsampling operator를 제거, 대신 연속하는 convolutional layer에 filter를 upsample
        - 더 높은 sampling rate를 가진 feature map을 출력
        - Filter upsampling는 zero가 아닌 값들로 이루어진 filter tap들 사이에서 hole(trous)을 삽입하는 것과 같다.
      - 실제로, atrous convolution의 결합으로 full resolution feature map을 회복
        - feature map을 더 dense하게 만듬
        - 기본 image size에서 feature response의 단순한 bilinear interpolation이 선행되어야 함
        - deconvolutional layer를 사용하는 것 보다 좋음
        - 기존 convolution과 비교해서 파라미터들의 개수나 계산의 양 증가 없이 filter의 field of view를 증가 시킴
  2. 다양한 크기에서 object의 존재 => Atrous Spatial Pyramid Pooling (ASPP)로 해결
    - 원인
      - 다양한 크기에서 object가 존재하는 것에 의해 발생
    - 해결
      - 일반적인 방법 : DCNN에서 제시됨.
        - 같은 이미지에서 크기를 조절한 여러가지 이미지를 만듬
        - feature랑 score map을 증가
        - 장점 : 시스템의 성능을 증가 
        - 단점 : 비용 (input image의 다양한 버전을 만들고, 모든 DCNN layer의 feature response를 계산해야 함)
      - convolution 전에 계산적으로 효율적인 스키마를 제시
        - spatial pyramid pooling에 영향을 받음.
        - 다른 sampling rate에서 multiple parallel atrous convolutional layer를 사용해서 이런 mapping을 실행
        - 제시된 이런 technique를 atrous spatial pyramid pooling이라고 부른다.
  3. DCNN invariance 때문에 localization accuracy 감소 => CRF로 해결
    - 원인
      - object를 중앙에 놓는 classifier는 spatial transformation에 invariance를 필요로 하게 된다.
      - 이는 본질적으로 DCNN의 spatial accuracy를 제한시킨다.
    - 해결
      - 일반적인 방법:
        - 마지막 segmentation 결과를 계산할 때 다양한 network로부터 "hyper-column" feature를 추출하기 위해서 skip-layer사용
      - 좀 더 효율적인 방법 : CRF를 사용
        - fully connected Conditional Random Field (CRF)를 적용시킴으로서 fine detail을 잡는 model의 능력을 향상시킴
        - 주로 semantic segmentation에서 사용
        - semantic segmentation에서 다양한 방법의 classifier들로 계산된 class score를 pixel들과 edge들, 혹은 superpixel들의 local interaction으로 잡아진 low-level information과 결합 
        - 효율적인 계산, fine dege detail을 잡을 능력, long range dependency 
        - pixel level classifier를 기반으로 한 성능을 향상
        
- Fig 1.에서 DeepLab model의 전반적인 모습 확인
[Fig1]
- VGG-16 이나 ResNet-101의 재 목적화 (image classification -> semantic segmentation)
  - fully connected layer를 convolutional layer로 변환 (즉, fully convolutional network)
  - atrous convolutional layer를 통한 feature resolution의 향상
    - 기본이 되는 network의 모든 32 픽셀 대신 모든 8 픽셀에 대한 응답을 계산할 수 있도록 한다.
  - bi-linear interpolation 적용
    - 기본 이미지 해상도에 도달하기 위해서 8개의 요소에 의해 score map을 upsample 하기 위함
    - segmentation 결과를 정제하는 fully-connected CRF로 input을 산출
- DeepLab 시스템의 3가지 장점
  1. Speed
    - atrous convolution의 장점으로 인해, DCNN은 NVidia Titan X GPU에서 8FPS로 운영된다.
      - fully-connected CRF를 위한 Mean Field Inference는 CPU에서 0.5초를 필요로 한다.
  2. Accuracy
    - PASCAL VOC 2012 semantic segmentation benchmark, PASCAL-Context, PASCAL-Person-Part, Cityscape에서 경쟁력 있는 점수를 얻음
  3. Simplicity
    - 잘 만들어진 모델 DCNNs과 CRFs의 cascade(계단식)로 구성되어 있다. 
  
코드 공유 : http://liangchiehchen.com/projects/DeepLab.html


