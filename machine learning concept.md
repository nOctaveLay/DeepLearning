# Machine learning concepts

## Index

1. [Motivation](#motivation)
    1. Prediction
    1. Inference
    
1. [Performance Analysis](#performance-analysis)
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
1. [Tuning](#tuning)
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
    
1. Types
    1. Regression
    
    1. Classification
    
    1. Clustering
    
    1. Density Estimation
    
    1. Dimensionality Reduction
    
1. Kind
    1. Parametic
    
    1. Non-Parametic
    
1. Categories
    1. Supervised
    
    1. Unsupervised
    
    1. Reinforcement Learning
    
1. Approaches
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
    
1. Taxonomy
    1. Generative Methods
        1. Mixtures of Gaussians, Mixtures of experts, Hidden Markov Models(HMM)
            
        1. Gaussians, Naïve Bayes, Mixtures of multinomials
            
        1. Sigmoidal belief networks, Bayesian networks, Markov random fields
            
    1. Discriminative Methods
        1. Logistic regression, SVMs
        1. Traditional neural networks, Nearest neighbor
        1. Conditional Random Fields (CRF)

1. Selection Criteria
    1. Prediction Accuracy vs Model Interpretability
    
1. Libraries
    1. Numpy
    
    1. Pandas
    
    1. Scikit-Learn
    
    1. Tensorflow
    
    1. MXNet
    
    1. Keras
    
    1. Torch
    
    1. Microsoft Cognitive Toolkit


## Motivation<a id="motivation"></a>
### Prediction

>When we are interested mainly in the predicted variable as a result of the inputs, <br>
>but not on the each way of the inputs affect the prediction. <br>
>In a real estate example, Prediction would answer the question of:<br> 
>Is my house over or under valued? <br>
>Non-linear models are very good at these sort of predictions, but not great for inference because the models are much less interpretable.<br>
<br>

우리가 input의 결과로서 예측된 variable을 주로 관심있어 할 때, 그렇지만 input의 각각의 방법이 예측에 영향을 끼치지 않을 때<br>
대표적인 예제 : 내 집의 가치가 높이 평가됬나요? 낮게 평가됬나요?<br>
Non-linear model이 이러한 예측에 매우 좋다. 그렇지만 inference에는 좋지 않다. 왜냐하면 모델을 해석하기 어렵기 때문이다.<br>

### Inference
>When we are interested in the way each one of the inputs affect the prediction. <br>
>In a real estate example, Inference would answer the question of: How much would my house cost if it had a view of the sea? <br>
>Linear models are more suited for inference because the models themselves are easier to understand than their non-linear counterparts<br><br>

우리가 input의 각각의 방법이 예측에 영향을 끼치는 것에 대해 궁금해 할 때<br>
대표적인 예제 : 내가 바닷가가 보이는 집을 가지려면, 얼마나 들까요?<br>
Linear model은 inference에 적합하다. 왜냐하면 모델이 Non linear counterpart보다 이해하기 쉽기 때문이다.<br>

## Performance Analysis<a id="performance-analysis"></a> 
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

Out of all the examples the classifier labeled as positive, what fraction were correct?
positive라고 이름 붙인 classifier있을 때, **정말로** positive한 fraction은 무엇일까?

#### Recall
> (TP)/ (TP+FN)
> 정말 병이 있기 때문에 병이 있다고 진단한 비율.
> 정말 병이 있는 사람들 중에서의 TP
> confusion matrix의 bottom row

Out of all the positive examples there were, what fraction did the classifier pick up?
정말로 모든 예제들이 positive example일 때 어떤 fraction을 고를 것인가?

### Harmonic Mean of Precision and Recall

(2 * p * r / (p+r))


### ROC Curve - Receiver Operating Characteristics
[roc-curve]
True Positive Rate (Recall / Sensitivity) vs False Positive Rate (1-Specificity)
정말 Positive한 비율 (Recall/ Sensitivity) vs 가짜 Positive한 비율 (1 - Specificity)

### Bias-Variance Tradeoff
Bias refers to the amount of error that is introduced by approximating a real-life problem, which may be extremely
complicated, by a simple model.<br>
= bias는 극단적으로 복잡하고, 단순한 model에 의한 현실 세계의 문제를 어림짐작함으로써 소개된 error의 양을 의미한다.

If Bias is high, and/or if the algorithm performs poorly even on your training data, try adding more features, or a more flexible model.
= 만약 Bias가 높다면, 그리고/또는 알고리즘이 training data를 동등히 나누지 못한다면, 더 많은 feature를 더하려 하거나 더 flexible한 모델을 쓰려 할 것이다.

Variance is the amount our model’s prediction would change when using a different training data set.
High: Remove features, or obtain more data.
= Variance는 다른 training data set을 사용했을 때 우리의 예측이 변화시키는 양이다.
높을 경우, feature를 제거하거나, 더 많은 data를 얻어야한다.

### Goodness of Fit = R^2
1.0 - sum of squared errors / total sum of squares(y)

### Mean Squared Error(MSE)
The mean squared error (MSE) or mean squared deviation (MSD) of an estimator (of a procedure for
estimating an unobserved quantity) measures the average of the squares of the errors or deviations—that
is, the difference between the estimator and what is estimated

MSE(=관측되지 않은 양을 측정하는 과정에서 나온 측정자의 MSD)는 estimator와 무엇이 측정되었는 가의 차이에서의 deviation 또는 error들의 square의 평균이다.

### Error Rate
The proportion of mistakes made if we apply out estimate model function the the training observations in a classification setting

classification setting에서 estimate model function을 훈련된 observation에 적용시킬 때 mistake의 비율


## Tuning
### Cross-validation


# 이미지 필요
[confusion Matrix]
[roc-curve]
