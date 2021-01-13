# An overview of gradient descent optimization algorithms

## 정보

- 원래 블로그 출처였다. [blog post](http://sebastianruder.com/optimizing-gradient-descent/index.html)
- 현재에도 블로그엔 업데이트가 지속중이다.
- 논문 [출처](https://arxiv.org/pdf/1609.04747.pdf)
- Optimization : object function을 optimize하는 것.

## Index

- [Abstract](##Abstract)
- [Introduction](##Introduction)
- [Gradient descent variants](##Gradient%20descent%20variants)
  - [Batch gradient descent](###1.%20Batch%20gradient%20descent)
  - [Stochastic gradient descent](###2.%20Stochastic%20gradient%20descent)
  - [Mini-batch gradient descent](###3.%20Mini-batch%20gradient%20descent)
- [Challenges](##Challenges)
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

- Section2 : gradient descent의 다양한 변형
- Section3 : 훈련을 하는 동안 생기는 문제점들을 간략하게 요약
- Section4 : 이러한 문제점들을 해결하기 위한 동기를 보여주면서, 가장 일반적인 최적화 알고리즘에 대한 소개
- Section5 : parallel 하고 distributed setting에서 gradient descent를 optimize하기 위해 algorithm과 architectur를 짧게 볼 것이다.
- Section6 : gradient descent를 최적화하는 것을 도와주는 추가적인 전략에 대해 고려해볼 것이다.

## Gradient descent variants

- gradient descent의 3가지 변형이 있다.
- accuracy of the parameter와 update를 수행하기 위해 걸리는 시간 사이엔 항상 trade-off가 있다.

### 1. Batch gradient descent

- Vanilla gradient descent = batch gradient descent라고 불린다.
- cost function의 gradient를 계산한다.
- 이는 다음과 같이 나타낼 수 있다.
- ![Batch Gradient Descnet](./images/batch_gradient_descent.PNG)
- 매우 느리고 메모리에 맞지 않는 dataset을 다루기 어려워한다.
- online에서(새로운 example로) model을 업데이트하는 것을 허용하지 않는다.
- 순서
  - loss function, data, parameter로 gradient를 구한다.
  - parameter = parameter - learning_rate * gradient
- 왠만한 딥러닝 라이브러리에는 그라이언트를 계산하기 편하도록 자동으로 미분해주는 기능이 있다.
- gradient를 직접 얻을 경우, gradient check를 하는 건 좋은 생각이다.
- [gradient check](https://cs231n.github.io/neural-networks-3/)
- parameter를 얼마나 우리가 업데이트를 크게 할지를 결정하는 learning rate를 가지고 gradient 방향으로 업데이트 한다.
- Batch gradient descent는 convex error surface를 위해 global minimum으로 수렴할 것을, non-convex surface일 때 local minimum으로 수렴할 것을 보장한다.
- Batch gradient descent는 큰 데이터 셋에서 중복 계산(redundant computation)을 한다.
  - 각각의 parameter가 update되기 전에 유사한 예제로 gradient를 재계산한다.

```python
for i in range ( nb_epochs ):
    params_grad = evaluate_gradient ( loss_function , data , params )
    params = params - learning_rate * params_grad
```

### 2. Stochastic gradient descent

- 각각의 training example마다 parameter를 update 하는 것.
- ![Stochastic Gradient Descent](./images/stochastic_gradient_descent.PNG)
- SGD는 한 타임에 하나만 update함으로서 중복 계산을 버린다.
- 훨씬 더 빠르고, online에서 학습시킬 수 있다.
- objective function이 무겁게 변동하게 만드는 높은 variance에서 빈번하게 update를 수행한다.
- ![Fig 1](./images/SGD_fluctuation.PNG)
- SGD의 fluctuation은 가능한 새롭고, 가능한 더 local minimum으로 jump할 수 있도록 한다.
  - 이것은 SGD가 overshooting을 계속 한다면 궁극적으로 추출된 minimum으로 수렴하는 것과 충돌이 난다.
- 우리가 learning rate를 천천히 줄일 때 SGD는 batch gradient descent와 동일하게 수렴하는 행동을 보여준다.
  - 거의 non-convex에서는 local minimum으로, convex에서는 global minimum으로 각각 수렴한다.

```python
for i in range ( nb_epochs ):
    np.random.shuffle ( data )
    for example in data :
        params_grad = evaluate_gradient ( loss_function , example , params )
        params = params - learning_rate * params_grad
```

- 반드시 data를 항상 shuffle한다는 것을 기억하라.

### 3. Mini-batch gradient descent

- n 개의 training example의 모든 mini-batch 단위로 update를 수행함.
- parameter update의 variance를 줄임.
  - 이는 수렴하는 값을 더 안정적으로 이끔
- 높게 최적화된 matrix optimization의 사용을 최신의 딥러닝 라이브러리에 일반적으로 만듬.
  - 이는 mini-batch의 gradient를 효율적으로 계산할 수 있게 해줌
- 일반적인 mini-batch size는 50 - 256이지만, 적용에 따라 달라질 수 있다.
- Mini-batch algorithm은 신경망을 훈련할 때 전형적인 선택의 알고리즘이다.
- SGD라는 용어도 mini-batch들이 사용됬을 때 적용되었다.
- batch size = 50 일 때

```python
for i in range ( nb_epochs ):
    np . random . shuffle ( data )
    for batch in get_batches ( data , batch_size =50):
        params_grad = evaluate_gradient ( loss_function , batch , params )
        params = params - learning_rate * params_grad
```

## Challenges