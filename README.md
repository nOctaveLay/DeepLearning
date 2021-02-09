# 머신러닝/딥러닝에 대해 정리한 내용

- 주로 논문을 알기 쉽게 정리합니다.
- 논문 요약과 논문 전체로 구성 되어 있습니다.
- 정리를 바라거나 읽고 싶은(읽어야 하는) 논문은 **정보 모음** 헤더에 정리됩니다.
  - 추후 다른 헤더로 옮겨질 내용들이 적혀있습니다.

## 딥러닝 기초

- [Deep Learning mindmap source 2020](https://whimsical.com/CA7f3ykvXpnJ9Az32vYXva)
  - [Machine Learning Data Processing](Basic%20Concept/MindMap/ML%20Process.md)
  - [Machine Learning Concept](Basic%20Concept/MindMap/ML%20concept.md)
  - [Machine Learning Process](Basic%20Concept/MindMap/ML%20process.md)
- [Optimization](https://arxiv.org/pdf/1609.04747.pdf) [[정리]](Basic%20Concept/Overview_of_Gradient_descent_optimization.md)
- ![define](./Basic%20Concept/images/define.png)

## Image Segmentation

### Classification

> Categorizing the entire image into a class such as "people", "animals", "outdoors"

- [Imagenet classification with deep convolutional neural networks[CNN, AlexNet]](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### Object Detection

> detecting objects within an image and drawing a rectangle around them, for example, a person or a sheep

### Segmentation

> classifies all the pixels of an image into meaningful classes of objects.
>
> These classes are “semantically interpretable” and correspond to real-world categories.
>
>For instance, you could isolate all the pixels associated with a cat and color them green. This is also known as dense prediction because it predicts the meaning of each pixel.

#### Semantic segmentation

> pixel-level classification

#### [Awesome Semantic Segmentation](https://github.com/mrgloom/awesome-semantic-segmentation?fbclid=IwAR35vXm16KQ_TG2C9361lreBhkYP82ZJioNI-UCyDdr0WpQhM_RBNVwZrPw)

- [U-net](https://arxiv.org/pdf/1505.04597.pdf) [[정리]](Image%20Segmantation/Semantic%20Segmentation/UNet.md)
- [SegNet](https://arxiv.org/pdf/1511.00561.pdf) [[정리]](Image%20Segmantation/Semantic%20Segmentation/SegNet.md)
- DeepLab
  - [DeepLabv1](https://arxiv.org/pdf/1412.7062.pdf) [[정리]](Image%20Segmantation/Semantic%20Segmentation/Deeplab/Deeplab_v1.md)
  - [DeepLabv2](https://arxiv.org/pdf/1606.00915.pdf) [[정리]](Image%20Segmantation/Semantic%20Segmentation/Deeplab/Deeplab_v2.md)
  - [DeepLabv3](https://arxiv.org/pdf/1706.05587.pdf) [[정리]](Image%20Segmantation/Semantic%20Segmentation/Deeplab/Deeplab_v3.md)
  - [DeepLabv3+](https://arxiv.org/pdf/1802.02611.pdf) [[정리]](Image%20Segmantation/Semantic%20Segmentation/Deeplab/Deeplab_v3+.md)
- R CNN
  - [R CNN](https://arxiv.org/abs/1311.2524) [[정리]](Image%20Segmantation/Semantic%20Segmentation/R%20CNN/R-CNN.md)
  - [Mask R CNN](https://arxiv.org/pdf/1703.06870.pdf) [[정리]](Image%20Segmantation/Semantic%20Segmentation/R%20CNN/Mask_R-CNN.md)
- Spatial pyramid:
  - S. Lazebnik, C. Schmid, and J. Ponce, “Beyond bags of features: Spatial pyramid matching for recognizing natural scene categories,” in CVPR, 2006.
  - K. He, X. Zhang, S. Ren, and J. Sun, “Spatial pyramid pooling in deep convolutional networks for visual recognition,” in ECCV, 2014

## NLP

1. [데이터 전처리](https://blog.pingpong.us/dialog-bert-tokenizer/?fbclid=IwAR0O2mtCrn4ilEusZE2fV3waGWl1BGE7Q3ifV6TBHu-nbQ5XViflE271B2U)
2. [딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155?fbclid=IwAR1jXkBthksuedED_dxANn_NRHzKgSw1oBEoZSPPnNJulpcDyUSg17BokG8)

---

1. [RoBERTa](https://arxiv.org/abs/1907.11692?fbclid=IwAR1ZISElXegapWYpz0Ut3kV3mQFoh8IOiJevKJd5QH9P7SMt9XJWKFfrgx4)
2. [ALBERT](https://arxiv.org/abs/1909.11942?fbclid=IwAR2TNdGL_aFnuB1x5e4YxhvnwfQcEgcjxFBmlFrQ8NGASu1nP1M09GgWZ4w)
3. [BERT](http://docs.likejazz.com/bert/?fbclid=IwAR2TNdGL_aFnuB1x5e4YxhvnwfQcEgcjxFBmlFrQ8NGASu1nP1M09GgWZ4w#fn:fn-2)

## 정보 모음

- [개인적으로 많은 정보가 들어있다고 생각하는 사이트](https://deep-learning-drizzle.github.io/?fbclid=IwAR2HVeEddlfF0WaEPW4IRRq6oUVtOp1BPcTNdGHABgaKvrhKJ7HzcW8GJVo
)
- [Deview 2019](https://deview.kr/2019/schedule)
- [Best paper awards in Computer Science](https://jeffhuang.com/best_paper_awards/?fbclid=IwAR1xqjapSTqkqGb_bi7qBaeTT5me8Jv8mUc2s6M6TzBVAfSzBovBYG8aotc)
- [Best Paper Awards at ACL 2020](https://acl2020.org/blog/ACL-2020-best-papers/)
- [Difference](https://missinglink.ai/guides/computer-vision/image-segmentation-deep-learning-methods-applications/)

## 기여

- README.md에 기여하실 경우
  - 만약 일치하는 항목이 없을 경우, header를 2개 붙여 새로운 헤더를 만듭니다. (ex 딥러닝 기초)
  - 모든 항목은 Header 안의 하위 항목으로 적으며, 반드시 해당 항목이 있는 위치로 이동할 수 있도록 해야합니다.
  - 모든 논문 정리들은 각 Header에 맞는 폴더로 이동되어야하며, 논문 정리에 들어갈 이미지들은 각 헤더 폴더 안에 images라는 폴더를 만들어 image를 넣어주세요.
  - 메인 Topic엔 출처 링크를, [정리]에는 논문 정리 링크(markdown)를 걸어줍니다.
  - 이 때 요약본과 전체본이 따로라면, [요약] [전체]로 각각 걸어줍니다.
  - 논문 전체 정리시에는 반드시 첫 번째 헤더엔 논문의 제목이 들어가야합니다.
  - 두 번째 헤더부터 논문에서 정리된 내용의 Index를 삽입하시면 됩니다.
  - (Recommanded) Index를 붙여주시면 나중에 다른 분들이 보시기 쉽습니다.
  - (Re`commanded) 딥러닝 기초엔 헤더에 대한 설명이 들어가면 더욱 좋습니다.

## 출처

- [분류에 대한 도움](https://missinglink.ai/guides/computer-vision/image-segmentation-deep-learning-methods-applications/)
