# Machine learning concepts

## Index
1. [Motivation] (#motivation)
    1. Prediction
    1. Inference
    
1. Performance Analysis
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
1. Tuning
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


## Motivation <a id="motivation"></a>
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

## Performance Analysis
### Confusion Matrix
[confusion Matrix]

### Accuracy
Fraction of correct predictions, not reliable as skewed when the data set is unbalanced (that is, when the number of samples in different classes vary greatly)



[confusion Matrix]
