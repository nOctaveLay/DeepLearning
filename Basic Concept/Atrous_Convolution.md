# Atrous Convolution

- 소개
  - [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](/Semantic%20Segmentation/Deeplab/Deeplab.md)
- 원인
  - DCNN 안의 연속된 layer에서 실행되는 max-pooling과 downsampling('striding')의 반복적인 결합으로 인해 발생.
  - fully convolutional fashion에서 DCNN이 적용됬을 때 감소된 spatial resolution을 가진 feature map을 만들어 냄.
- 해결
    - 마지막 부분에 존재하는, 몇 개의 max pooling layer 안의 downsampling operator를 제거, 대신 다음의 convolutional layer안에서 filter를 upsample
    => 더 높은 sampling rate를 가진 feature map을 산출한다.
    - Filter upsampling을 할 때 zero가 아닌 값들로 이루어진 filter tap들 사이에서 hole(trous)을 삽입한다.
    - 이 방법은 undecimated wavelet transform에서 효율적인 계산을 위해 고안된 방법이다.
    - 이를 atrous convolution으로 부르기로 한다.
    - 실제로, atrous convolution을 조합함으로서 full resolution feature map을 회복한다.
    - 이는 feature map을 더 조밀하게 만듬
    - atrous convolution은 기본 image size에서 feature response의 단순한 bilinear interpolation으로 이어진다.
    - dense prediction을 하는 데 있어, deconvolutional layer를 사용하는 것보다 단순하고, 강하다.
    - 큰 필터를 가지고 있는 정규 convolution과 비교하면, atrous convolution은 파라미터들의 개수나 computation의 양이 증가 하는 일 없이 filter의 시야를 효과적으로 증가시킬 수 있다.