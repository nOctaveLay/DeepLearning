# Atrous Convolution

- 소개
  - [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](/Semantic%20Segmentation/Deeplab/Deeplab.md)
  - Atrous convolution: Filter upsampling을 할 때 zero가 아닌 값들로 이루어진 filter tap들 사이에서 hole(trous)을 삽입하는 것.
  - 이 방법은 undecimated wavelet transform에서 효율적인 계산을 위해 고안된 방법이다.
  - ![atrous convolution](/Semantic%20Segmentation/Deeplab/images/DeepLab_v1_atrous_convolution.PNG)
- 계기
  - fully convolutional fashion에서 DCNN이 적용됬을 때 DCNN 안의 연속된 layer에서 실행되는 max-pooling과 downsampling('striding')의 반복적인 결합으로 인해 signal down-sampling 발생
  - 즉, 감소된 spatial resolution을 가진 feature map을 만들어 냄.
  - 이를 해결하기 위해 Atrous Convolution이 제시됨.

- Signal down-sampling
  - 해결하는 방안
    - 마지막 부분에 존재하는, 몇 개의 max pooling layer 안의 downsampling operator를 제거, 대신 다음의 convolutional layer안에서 filter를 upsample해서 더 높은 sampling rate를 가진 feature map 산출.
    - Filter upsampling을 할 때 zero가 아닌 값들로 이루어진 filter tap들 사이에서 hole(trous)을 삽입.

  - 결과
    - 일반적으로 적용되고, 어떤 target sampling rate든 간에 어떤 aprroximation들을 소개하는 일 없이 dense CNN feature map을 효과적으로 계산할 수 있다.
    - Atrous convolution을 조합함으로서 full resolution feature map을 회복함.
    - 이는 feature map을 더 조밀하게 만듬
    - atrous convolution은 기본 image size에서 feature response의 단순한 bilinear interpolation으로 이어진다.
    - dense prediction을 하는 데 있어, deconvolutional layer를 사용하는 것보다 단순하고, 강하다.
    - 큰 필터를 가지고 있는 정규 convolution과 비교하면, atrous convolution은 파라미터들의 개수나 computation의 양이 증가 하는 일 없이 filter의 field-of-view를 효과적으로 증가시킬 수 있다.
