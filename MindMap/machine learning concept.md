# Machine learning concepts

## Index

1. [Motivation](#motivation)<a id="moti-top"></a>
    1. Prediction
    1. Inference
    
1. [Performance Analysis](#performance-analysis)<a id="perfo-top"></a>
    1. Confusion Matrix
    1. Accuracy
    1. f1 score
        1. Precision
        1. Recall
        1. Harmonic Mean of Precision and Recall
    1. ROC Curve - Receiver Operating Characteristics
    1. Bias-Variance Tradeoff
    1. Goodness of Fit = R^2 
    1. Mean Squared Error
    1. Error Rate 
1. [Tuning](#tuning)<a id="tune-top"></a>
    1. Cross-validation
        1. Leave-p-out cross-validation
        1. Leave-one-out cross validation
        1. k-fold cross-validation
        1. Holdout method
        1. Repeated random sub-sampling validation  
    1. Hyperparameters
        1. Grid Search
        1. Random Search
        1. Gradient-based optimization
    1. Early Stopping(Regularization)
    1. Overfitting
    1. Underfitting
    1. Bootstrap
    1. Bagging
1. [Types](#types)<a id="types-top"></a>
    1. Regression
    1. Classification
    1. Clustering
    1. Density Estimation
    1. Dimensionality Reduction
1. [Kind](#kind)<a id = "kind-top"></a>
    1. Parametic
    1. Non-Parametic
1. [Categories](#categories)<a id="cate-top"></a>
    1. Supervised
    1. Unsupervised
    1. Reinforcement Learning
1. [Approaches](#approaches)<a id="appro-top"></a>
1. [Taxonomy](#taxonomy)<a id="tax-top"></a>
    1. Generative Methods
        1. Mixtures of Gaussians, Mixtures of experts, Hidden Markov Models(HMM)  
        1. Gaussians, Naïve Bayes, Mixtures of multinomials
        1. Sigmoidal belief networks, Bayesian networks, Markov random fields
    1. Discriminative Methods
        1. Logistic regression, SVMs
        1. Traditional neural networks, Nearest neighbor
        1. Conditional Random Fields (CRF)
1. [Selection Criteria](#selection-criteria)<a id="selc-top"></a>
    1. Prediction Accuracy vs Model Interpretability
1. [Libraries](#libraries)<a id="lib-top"></a>



## [Motivation](#moti-top)<a id="motivation"></a>
### Prediction

>When we are interested mainly in the predicted variable as a result of the inputs, <br>
>but not on the each way of the inputs affect the prediction. <br>
>In a real estate example, Prediction would answer the question of:<br> 
>Is my house over or under valued? <br>
>Non-linear models are very good at these sort of predictions, but not great for inference because the models are much less interpretable.<br>
<br>

* 우리가 input의 결과로서 예측된 variable을 주로 관심있어 할 때, 그렇지만 input의 각각의 방법이 예측에 영향을 끼치지 않을 때 사용<br>
* 대표적인 예제 : 내 집의 가치가 높이 평가됬나요? 낮게 평가됬나요?<br>
* Non-linear model O but inference x <- 모델 해석 어려움.<br>

### Inference
>When we are interested in the way each one of the inputs affect the prediction. <br>
>In a real estate example, Inference would answer the question of: How much would my house cost if it had a view of the sea? <br>
>Linear models are more suited for inference because the models themselves are easier to understand than their non-linear counterparts<br><br>

* 우리가 input의 각각의 방법이 예측에 영향을 끼치는 것에 대해 궁금해 할 때<br>
* 대표적인 예제 : 내가 바닷가가 보이는 집을 가지려면, 얼마나 들까요?<br>
* Linear model은 inference에 적합 <- 모델이 Non linear counterpart보다 이해하기 쉬움<br>

## [Performance Analysis](#perfo-top)<a id="performance-analysis"></a> 
### Confusion Matrix
[confusion Matrix]

### Accuracy
Fraction of correct predictions, not reliable as skewed when the data set is unbalanced (that is, when the number of samples in different classes vary greatly)<br><br>

* 맞는 예측들의 조각, dataset이 unbalanced되어 있을 때 비뚤어져 있기 때문에 신뢰할 수 있는 게 아님.
* 다른 샘플의 수가 매우 다양하게 있음

### f1 score
#### Precision
> (TP) / (TP + FP)
> 암이 있다고 진단한 사람이 실제로 암이 있을 확률.
> 다른 말로 말하자면, 암이 있다고 진단한 집단 중에서 TP의 비율.

Out of all the examples the classifier labeled as positive, what fraction were correct?<br>
* positive라고 이름 붙인 classifier가 있을 때, **정말로** positive한 fraction은 무엇일까?<br><br>

#### Recall
> (TP)/ (TP+FN)
> 정말 병이 있기 때문에 병이 있다고 진단한 비율.
> 정말 병이 있는 사람들 중에서의 TP
> confusion matrix의 bottom row

>Out of all the positive examples there were, what fraction did the classifier pick up?<br>
* 정말로 모든 예제들이 positive example일 때 어떤 fraction을 고를 것인가?<br>

### Harmonic Mean of Precision and Recall

(2 * p * r / (p+r))


### ROC Curve - Receiver Operating Characteristics
[roc-curve]
True Positive Rate (Recall / Sensitivity) vs False Positive Rate (1-Specificity)<br>
정말 Positive한 비율 (Recall/ Sensitivity) vs 가짜 Positive한 비율 (1 - Specificity)<br>

### Bias-Variance Tradeoff
>Bias refers to the amount of error that is introduced by approximating a real-life problem, which may be extremely
complicated, by a simple model.<br>
>If Bias is high, and/or if the algorithm performs poorly even on your training data, try adding more features, or a more flexible model.<br>

- bias는 현실 세계의 문제를 어림짐작 -> 그 차이로 발생하는 error량<br>
- Bias 높음 & 알고리즘이 training data를 동등히 나눔 x -> 많은 feature 더함 & flexible한 모델 사용<br><br>

>Variance is the amount our model’s prediction would change when using a different training data set.
>High: Remove features, or obtain more data.<br>
= Variance는 다른 training data set을 사용했을 때 우리의 예측이 변화시키는 양이다.<br>
높을 경우, feature를 제거하거나, 더 많은 data를 얻어야한다.<br><br>

### Goodness of Fit = R^2
1.0 - sum of squared errors / total sum of squares(y)<br><br>

### Mean Squared Error(MSE)
>The mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors or deviations—that is, the difference between the estimator and what is estimated<br><br>

- MSE(=관측되지 않은 양을 측정하는 과정에서 나온 측정자의 MSD)
- estimator와 측정된 것의 차이에서 생기는 deviation, error들의 square 평균<br><br>

### Error Rate
>The proportion of mistakes made if we apply out estimate model function the the training observations in a classification setting<br><br>

* classification setting에서 estimate model function을 훈련된 observation에 적용시킬 때 mistake의 비율<br><br>


## [Tuning](#tune-top)<a id="tuning"></a>
### Cross-validation
>One round of cross-validation involves partitioning a sample of data into complementary subsets, performing the analysis on one subset (called the training set), and validating the analysis on the other subset (called the validation set or testing set). <br>
>To reduce variability, multiple rounds of cross-validation are performed using different partitions, and the validation results are averaged over the rounds<br><br>

- cross-validation의 한 라운드 : <br>
    - 하나의 subset (= training set) 분석 & 다른 subset(= validation set or testing set) validating.<br>
    - data sample -> complementary set 으로 분리<br><br>
- variability 줄임 <br>
    - 각기 다른 파티션을 사용하면서 많은 cross validation 라운드가 돔<br>
    - 이 라운드마다 validation result 평균을 냄.<br>
- 종류 <br>
1. Leave-p-out cross validation
2. Leave-one-out cross validation
3. k-fold cross-validation
4. Holdout method
5. Repeated random sub-sampling validation

### Hyperparameters

#### Grid Search

>The traditional way of performing hyperparameter optimization has been grid search, or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. <br>
>A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set. <br><br>
        
* Hyperparameter 최적화 또는 파라미터 sweep의 전통적인 방식.<br>
    - 학습 알고리즘의 하이퍼 파라미터 공간의 수작업으로 지정된 subset을 통한 exhaustive searching이다.<br>
* 어떤 performance metric으로 가이드됨.<br>
    - 특히 training set에 cross-validation으로 측정 또는 held out validation set을 평가함으로서 측정<br><br>
             
#### Random Search
>Since grid searching is an exhaustive and therefore potentially expensive method, several alternatives have been proposed. <br>
>In particular, a randomized search that simply samples parameter settings a fixed number of times has been found to be more effective in high-dimensional spaces than exhaustive search.<br><br>
    
* grid search가 너무 시간이 많이 많이 걸림 -> 새로운 방법 필요
* randomized search는 고정된 시간동안 exhaustive search 보다 high-dimensional space에서 더 효과적이라고 밝혀진다.

#### Gradient-based optimization
>For specific learning algorithms, it is possible to compute the gradient with respect to hyperparameters and then optimize the hyperparameters using gradient descent. <br>
>The first usage of these techniques was focused on neural networks. <br>
>Since then, these methods have been extended to other models such as support vector machines or logistic regression<br><br>

- 대부분의 hyperparameter의 경우, gradient를 계산하는 것이 가능하다. -> 따라서 gradient descent를 이용해 최적화한다.<br>
- 신경망에서 처음 쓰였다.<br>
- support vector machine이나 logistic regression같은 다른 모델로 확장된다.<br><br>

### Early Stopping(Regularization)
>Early stopping rules provide guidance as to how many iterations can be run before the learner begins to over-fit, and stop the algorithm then.<br><br>

- learner가 over-fit을 시작하기 전에 얼마나 많은 iteration이 돌아야 하는 지에 대해 가이드라인을 제공.
- 그 다음 알고리즘을 멈춤

### Overfitting 
>When a given method yields a small training MSE (or cost), but a large test MSE (or cost), we are said to be
overfitting the data. <br>
>This happens because our statistical learning procedure is trying too hard to find pattens in the data, that might be due to random chance, rather than a property of our function. <br>
>In other words, the algorithms may be learning the training data too well. <br>
If model overfits, try removing some features, decreasing degrees of freedom, or adding more data.<br><br>

- 주어진 method가 samll training MSE이지만 큰 test MSE를 할 때, data의 overfitting이 이루어진다고 말한다.<br>
- 통계학적인 학습 방법 -> 너무 패턴을 찾기 어려워 -> overfitting 일어남<br>
- function의 property대신 random chance이기 때문에 일어남<br>
- 너무 잘 학습해서 일어나는 문제<br>
- overfitting이 일어났을 경우 = feature 지움, freedom한 정도를 지움, data를 더 추가함 <br><br>

### Underfitting
>Opposite of Overfitting. <br>
>Underfitting occurs when a statistical model or machine learning algorithm cannot capture the underlying trend of the data. <br>
>It occurs when the model or algorithm does not fit the data enough. <br>
>Underfitting occurs if the model or algorithm shows low variance but high bias (to contrast the opposite, overfitting from high variance and low bias). <br>
>It is often a result of an excessively simple model.<br><br>

- Overfitting의 반대<br>
- Underfitting: statistical model/machine learning algorithm -> data의 방향성을 capture x
- data 충분치 못할 때 발생<br>
- low variance 그러나 high bias를 보여줄 때 일어남. (overfitting은 high variance와 low bias를 보여줄 때 일어남)<br>
- 과도하게 심플한 모델의 결과.<br><br>

### Bootstrap
>Test that applies Random Sampling with Replacement of the available data, and assigns measures of accuracy (bias,
variance, etc.) to sample estimates<br>
1. Random Sampling -> 사용 가능한 data의 배치 적용 -> test<br>
2. 정확도 측정 -> sample estimate에 적용 -> test<br>

### Bagging
>An approach to ensemble learning that is based on bootstrapping. <br>
>Shortly, given a training set, we produce multiple different training sets (called bootstrap samples), by sampling with replacement from the original dataset. 
>Then, for each bootstrap sample, we build a model. <br>
>The results in an ensemble of models, where each model votes with the equal weight. <br>
>Typically, the goal of this procedure is to reduce the variance of the model of interest (e.g. decision trees).<br><br>

- bootstrapping에 기초한 ensemble learning에 대한 접근<br>
- training set이 주어지면, original dataset에서 replacement로 sampling함으로서 다양한 training set을 생성 <br>
- 결과는 model의 ensemble이고, 각각의 model은 동등한 weight를 가진다.<br>
- 전형적으로, 이 과정의 목표는 흥미로운 모델의 variance를 줄이는 것.(ex decision trees)<br>

## [Types](#types-top)<a id="types"></a>
- Regression : supervised problem, 결과는 discrete하기보단 continuous하다.<br>
- Classification<br>
    - input : 2개 혹은 이상의 클래스로 나뉨<br>
    - learner : 이런 클래스에 대해 보이지 않는 input 값을 할당하는 모델 만들어야 함.<br>
    - supervised learning으로 다뤄짐<br>
- Clustering<br>
    - input의 집합이 group으로 나누어짐<br>
    - group에 대해 미리 알 필요는 없음 (classification은 미리 알아야함)<br>
    - unsupervised learning으로 다뤄짐<br>
- Density Estimation<br>
    - input의 분포도를 어떤 공간에서 찾는 것<br>
- Dimensionality Reduction <br>
    - 낮은 차원 공간으로 매핑시킴으로서 input을 단순화 시키는 것. <br>
    
 ## [Kind](#kind-top)<a id="kind"></a>
 1. Parametric
    1. Making an assumption about the functional form or shape of our function (f),<br>
        - i.e: f is linear, thus we will select a linear model<br>
    2.  Selecting a procedure to fit or train our model. <br>
    This means estimating the Beta parameters in the linear function.<br> 
    A common approach is the (ordinary) least squares, amongst others<br><br>
    
    1. 함수의 모양이나 기능적인 형태에 대한 가정을 함(f) -> f가 linear 하니까, linear model 선택<br>
    2. 모델에 맞는 방법이나 모델을 훈련시킬 방법 선택<br>
        - 이 방법은 linear function의 Beta parameter를 추정하는 것이다.<br>
        - 일반적인 방법은 다른 것들 중에서 매우 작은 square<br>
 2. Non-Parametric<br>
When we do not make assumptions about the form of our function (f).<br>
However, since these methods do not reduce the problem of estimating f to a small number of parameters, a large number of observations is required in order to obtain an accurate estimate for f. An example would be the thin-plate
spline model.<br><br>

* 함수의 형태에 대해 가정하지 않을 때.
* 그렇지만, 이러한 방법이 추정하는 f의 문제를 적은 수의 parameter로 줄일 수 없을 때.
* 수많은 observation이 f를 위한 정확한 추정을 얻기 위해 필요됨
* thin-plate spline model

## [Categories](#cate-top)<a id="categories"></a>
1. Supervised
The computer is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs. <br>
- example input과 원하는 output이 제공됨.<br>
- 일반적인 input과 output의 rule을 알기 위함.<br>
<br>
2. Unsupervised
No labels are given to the learning algorithm, leaving it on its own to find structure in its input. Unsupervised learning can be a goal in itself (discovering hidden patterns in data) or a means towards an end (feature learning).<br>
- 알고리즘에 어떤 label도 없는 것. => 스스로 input을 보고 구조를 찾아내도록 놔두는 것<br>
- 목표 그 자체거나(data의 숨겨진 패턴을 파악), 목표로 향하는 수단을 학습하기를 원함(특징 학습 : feature learning)<br><br>

3. Reinforcement Learning<br>
A computer program interacts with a dynamic environment in which it must perform a certain goal (such as driving a vehicle or playing a game against an opponent). <br>
The program is provided feedback in terms of rewards and punishments as it navigates its problem space.<br>

- 특정한 목표를 수행해야 하는 dynamic environment와 상호작용할 때.<br>
- 프로그램은 reward와 punishment로 피드백을 받는다. -> problem space로 안내<br><br>

## [Approaches](#appro-top)<a id="approaches"></a>
1. Decision tree learning
1. Association rule learning
1. Artificial neural networks
1. Deep learning
1. Inductive logic programming
1. Support vector machines
1. Clustering
1. Bayesian networks
1. Reinforcement learning
1. Representation learning
1. Similarity and metric learning
1. Sparse dictionary learning
1. Genetic algorithms
1. Rule-based machine learning
1. Learning classifier systems

## [Taxonomy](#tax-top)<a id="taxonomy"></a>
1. Generative Methods
    1. Model class-conditional pdfs and prior probabilities. “Generative” since sampling can generate synthetic data points.
        - 모델의 class-conditional pdfs 이고, prior probabilities (이전 확률)이다.<br>
        - "Generative" 하다 = synthetic data point를 생성한다.<br>
    2. Popular Models
        1. Mixtures of Gaussians, Mixtures of experts, Hidden Markov Models(HMM)
        2. Gaussians Naive Bayes, Mixtures of multinomials
        3. Sigmoidal belief networks, Bayesian networks, Markov random fields
1. Discriminative Methods<br>
    * Directly estimate posterior probabilities. 
    * No attempt to model underlying probability distributions. 
    * Focus computational resources on given task– better performance<br><br>
    
    * 다음에 올 확률을 직접적으로 측정<br>
    * probability distribution에 놓여있는 모델에 어떠한 시도도 하지 않음<br>
    * 계산해야하는 자원을 주어진 일에 더 특화된 성능에 넣음.<br>
    <br>
2. Popular Models<br>
    1. Logistic regression, SVMs<br>
    2. Traditional neural networks, Nearest neighbor<br>
    3. Conditional Random Fields (CRF)<br><br>

## [Selection Criteria](#selc-top)<a id="selection-criteria"></a>
**Prediction Accuracy vs Model Interpretability**
>There is an inherent tradeoff between Prediction Accuracy and Model Interpretability, that is to say that as the model get more flexible in the way the function (f) is selected, they get obscured, and are hard to interpret.<br>
>Flexible methods are better for inference, and inflexible methods are preferable for prediction.<br><br>

<br>
- Prediction Accuracy(예측 정확도)와 Model Interpretability(해석도) 사이에선 내부적인 tradeoff O
    - 모델이 flexible 해질 때 -> 해석 어려움, 예측 쉬움

## [Libraries](#lib-top)<a id="libraries"></a>
* **Python**
    1. **Numpy**
    >Adds support for large, multi-dimensional arrays and matrices, along with a large library of high-level mathematical functions to operate on these arrays
    - 높은, 다중 차원의 array와 metrices를 지원.
    - 이런 array를 다루기 위해서 높은 수준의 수학적 함수들이 많음.<br><br>
    2. **Pandas**
    >Offers data structures and operations for manipulating numerical tables and time series
    - numerical table들과 time series를 다루기 위해서 data structure와 operation을 제공.<br><br>
    3. **Scikit-Learn**
    >It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy<br><br>
    - 다양한 classification, regression, clustering 알고리즘을 제공.
      = support vector machine, random forests, gradient booting, k-means, DBSCAN
    - Python numerical과 과학적인 libraries인 NumPy, SciPy와 상호작용을 하게끔 만들어져 있음<br><br>
    4. **Tensorflow**
    [tensorflow]<br><br>
    5. **MXNet**
    >Is an modern open-source deep learning framework used to train, and deploy deep neural networks. <br>
    >MXNet library is portable and can scale to multiple GPUs and multiple machines. <br>
    >MXNet is supported by major Public Cloud providers including AWS and Azure. <br>
    >Amazon has chosen MXNet as its deep learning framework of choice at AWS.<br><br>
    - open-source deep learning framework
    - DNN을 전개
    - 많은 GPU와 많은 machine을 가지고 다닐 수 있거나 확장가능하다.
    - 주로 AWS와 Azure를 포함한 Public Cloud provider가 지원한다.<br><br>
    6. **Keras**
    >Is an open source neural network library written in Python. <br>
    >It is capable of running on top of MXNet, Deeplearning4j, Tensorflow, CNTK or Theano. <br>
    >Designed to enable fast experimentation with deep neural networks, it focuses on being minimal, modular and extensible.<br><br>
    - Python으로 쓰여진 open source neural network library이다.
    - MXNet, Deeplearning4j, Tensorflow, CNTK 또는 Theano 위에서 돌 수 있다.
    - DNN에 빠른 실험을 할 수 있도록 고안됬다.
    - 작고, modular(모듈이 되고) 하고, extensible(확장)할 수 있다.<br><br>
    7. **Torch**
    >Torch is an open source machine learning library, a scientific computing framework, and a script language based on the Lua programming language. <br> 
    >It provides a wide range of algorithms for deep machine learning, and uses the scripting language LuaJIT, and an underlying C implementation.<br><br>
    - open source machine learning library이다.
    - 과학적으로 계산된 framework
    - Lua로 짜여진 script language이다.
    - deep machine learning에 대해 넓은 범위의 알고리즘 제공
    - LuaJIT라고 불리는 스크립트 언어를 사용
    - 기본적으로 C 실행을 한다.<br><br>
    8. **Microsoft Cognitive Toolkit**
    >Previously known as CNTK and sometimes styled as The Microsoft Cognitive Toolkit, is a deep learning framework developed by Microsoft Research. <br>
    >Microsoft Cognitive Toolkit describes neural networks as a series of computational steps via a directed graph.<br><br>
    - CNTK라고 알려진 것, 때때로 The Microsoft Cognitive Toolkit이라고 스타일되어있다.
    - Microsoft Research에서 개발한 deep learning framework
    - neural network를 directed graph를 통한 computational step의 연속이라고 묘사
    
    
# 이미지 필요
[confusion Matrix]
[roc-curve]
[tensorflow]
