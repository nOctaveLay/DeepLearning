# An overview of gradient descent optimization algorithms

## 정보

- 원래 블로그 출처였다. [blog post](http://sebastianruder.com/optimizing-gradient-descent/index.html)
- 현재에도 블로그엔 업데이트가 지속중이다.
- 논문 [출처](https://arxiv.org/pdf/1609.04747.pdf)

## Index

- Gradient descent variants
  - Batch gradient descent
  - Stochastic gradient descent
  - Mini-batch gradient descent
- Challenges
- Gradient descent optimization algorithms
  - Momentum
  - Nesterov accelerated gradient
  - Adagrad
  - Adadelta
  - RMSprop
  - Adam
  - AdaMax
  - Nadam
  - AMSGrad
  - Other recent optimizers
  - Visualization of algorithms
  - Which optimizer to use?
- Parallelizing and distributing SGD
  - Hogwild!
  - Downpour SGD
  - Delay-tolerant Algorithms for SGD
  - TensorFlow
  - Elastic Averaging SGD
- Additional strategies for optimizing SGD
  - Shuffling and Curriculum Learning
  - Batch normalization
  - Early Stopping
  - Gradient noise
- Conclusion
- References

## Abstract

- Gradient descent optimization algorithm은 매우 유명하지만, 그들의 강점과 약점을 실제적으로 설명하기 어렵기 때문에 black-box optimizer로 종종 사용된다.
- 이 article은 그들이 사용할 수 있도록 다른 알고리즘의 작동에 관련한 본질을 설명한다.
- 이 overview에서 보게 될 것
  - gradient descent의 다양한 variant들을 보게 될 것이다.
  - challenge들을 정리할 것이다.
  - 가장 일반적인 최적화 알고리즘을 설명할 것이다.
  - parallel and distributed setting에서의 architecture을 review할 것이다.
  - gradient descent를 최적화하는 데에 있어 추가적인 전략을 조사할 것이다.

## Introduction
