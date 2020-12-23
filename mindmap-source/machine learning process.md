# Machine learning process

## index
1. Question
    1. Classification
    1. Regression
    1. Anomaly Detection
    1. Clustering
    1. Reinforcement Learning
1. Direction
    1. SaaS
    1. Data Science and Applied Machine Learning
    1. Machine Learning Research 
1. Data
1. Model
1. Cost Function
1. Optimization
1. Tuning
1. Results and Benchmarking
1. Scaling
1. Deployment and Operationalisation
1. Infrastructure

## Question
1. Classification : Is this A or B?
2. Regression : How much, or how many of these?
3. Anomaly Detection : Is this anomalous?
4. Clustering : How can these elements be grouped?
5. Reinforcement Learning : What should I do now?

## Direction
1. SaaS - Pre-built Machine Learning models
    1. Google Cloud
        - Vision API
        - Speech API
        - Jobs API
        - Video Intelligence API
        - Language API
        - Translation API
    1. AWS
        - Rekognition
        - Lex
        - Polly
    1. 그 외 여러가지
1. Data Science and Applied Machine Learning
    1. Google Cloud - ML Engine
    1. AWS - Amazon Machine Learning
    1. Tools - Jupiter/Datalab/Zeppelin
    1. 그 외 여러가지
1. Machine Learning Research
    1. Tensorflow
    1. MXNet
    1. Torch
    1. 그 외 여러가지
    
### Data
1. Find
1. Collect
1. Explore
1. Clean Features
1. Impute Features
1. Engineer Features
1. Select Features
1. Encode Features
1. Build Datasets
    - Machine Learning = math (Linear Algebra)
    - data가 반드시 numeric해야 한다.

### Model
>Select Algorithm based on question and data available<br>
* question에 기반을 둔 Algorithm을 선택, 그리고 가능한 data 선택<br>

### Cost Function
>The cost function will provide a measure of how far my algorithm and its parameters are from accurately representing my training data.<br>
>Sometimes referred to as Cost or Loss function when the goal is to minimise it, or Objective function when the goal is to maximise it.<br>

* 얼마나 나의 알고리즘과 그 parameter가 training data에서 멀리 떨어져 있는지에 대한 측정을 알려준다.
* 때때로 목표가 이걸 최적화하기 위한 것이라면 Cost 또는 Loss function이라고도 말한다.<br>

### Optimization
>Having selected a cost function, we need a method to minimise the Cost function, or maximise the Objective function.
>Typically this is done by Gradient Descent or Stochastic Gradient Descent.<br>

* Cost function을 최소화하는 것, 혹은 Objective function을 최대화하는 것.
* Gradient Descent 혹은 Stochastic Gradient Descent로 구함
<br>
### Tuning
Different Algorithms have different Hyperparameters, which will affect
the algorithms performance. There are multiple methods for
Hyperparameter Tuning, such as Grid and Random search.
### Results and Benchmarking
