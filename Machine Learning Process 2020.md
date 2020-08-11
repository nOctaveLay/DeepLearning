
# Index

### [Data collection](#data-collection)
1. [Question to ask](#question-to-ask)
2. [Types of data](#type-data)
### [Data preparation](#data-pre)
1. [Exploratory data analysis (EDA), learning about the data you're working with ](#eda)
2. [Data preprocessing, preparing your data to be modelled](#data-pre-processing)
3. [Data splitting](#data-split)
### [Train model on data](#train-model)
1. [choose an algorithm](#choose-algorithm)
1. [Type of learning](#type-of-learning)
1. [Hyperparameter Tuning](#hyper-tuning)
### [Analysis/Evaluation](#analysis)
### [Serve model (deploying a model)](#serve-model) 
### [Retrain model](#retrain-model)
_ _ _

## Data Collection <a id="data-collection"></a>
### Question to ask <a id="question-to-ask"></a>
* What kind of problem are we trying to solve? (see machine learning problems)
* What data sources already exist?
* What privacy concerns are there?
* Is the data public?
* Where should we store the data?

### Types of data <a id="type-data"></a>
#### Structured data
- appears in tabulated format
- rows and columns style
- Can contain different types of data
    - Nominal/categorical : One thing or another, Order X
    - Numerical : Any continuous value where the difference between them matters.
    - Ordinal : Data which has order, but the distance between values is unknown.
        - 1-5 rate, 5 is not 5 times as good as 1
    - Time series : Data across time
        - 2012 - 2018 data
#### Unstructed data
- no rigid structure
- image, video, natural language text, speech

## Data Preparation<a id = "data-pre"></a>
### Exploratory data analysis (EDA), learning about the data you're working with <a id = "eda"></a>
- What are the feature variables(input) and the target variables(output)
    - have a disease or not
- What kid of data do you have? 
    - Create a data dictionary for what each feature is.
- Are there missing values? 
    - Should you remove them or fill them with feature imputation
- Where are the outliers?
    - How many of them are there?
    - Are they out by much (3+ standard deviations)?
    - Why are they there?
- Are there questions you could ask a domain expert about the data?
### Data preprocessing, preparing your data to be modelled<a id = "data-pre-processing"></a>
- Feature imputation(특징 삽입) : filling missing values 
    - Single imputation : Fill with mean, median of column.
    - Multiple imputation : Model other missing values and fill with what your model finds.
    - KNN : Fill data with a value from another example which is similar.
    - Many more, such as, random imputation, last observation carried forward, moving window, most frequent
- Feature encoding (value -> numbers, all value numerical.)
    - OneHotEncoding 
      - 모든 고유 벡터를 0 또는 1로 바꾸는 것
      - target value를 1로, 그 나머지를 0으로 한다.
      - ex) red, green, blue를 판별하고, 이게 red인 경우 [1,0,0]
    - LabelEncoder
      - label을 distinct numerical values로 바꾸는 것.
      - dog, cat, bird가 있으면 이는 각각 0, 1, 2로 된다.
    - Embedding encoding
      - 모든 다른 data point를 통해 representation을 배우는 것.
      - 예를 들면, 언어 모델은 단어들이 얼마나 가까이 있는지에 대한 표현으로 나타낸다.
      - Embedding은 structed data에 많이 쓰인다.
- Feature normalization (scaling) or standardization
    - numerical variables -> 전부 다른 scale
    - 어떤 머신러닝 알고리즘은 잘 수행하지 못한다.
    - scaling 과 standardization이 이걸 고치는 데 도움을 준다.
    - 종류
      - Feature scaling
        - normalization이라고 부른다.
        - value를 옮겨서 0-1사이에 위치하도록 만든다.
        - min value를 빼고, max - min으로 나눠서 만든다.
      - Feature standardization
        - 모든 value를 standard하게 만든다.
        - 평균이 0이 되고, unit variance를 갖게 만든다.
        - 특정 특징에 standard diviation(표준 편차)로 나누고 평균을 뺌으로서 만든다.
- Feature engineering
    - data -> 더 의미있는 표현으로 바꿈 (domain knowledge 첨가)
    - 종류
      - Decompose
        - 날짜를 분해 (2020-06-18을 2020,06,18 등으로 분해)
      - Dicretization
        - 큰 그룹을 작은 그룹으로 만듬
        - For numerical variable : 50개 이상 묶고, 50개 이하 묶고 이런걸 하는 걸 binning한다고 부름 (putting data into different "bins")
        - For categorical variables : light green color, dark green color -> green color 라고 부르는 것
      - Crossing and interaction features
        - 2개 혹은 그 이상의 특징을 결합시키는 방법
        - 두개의 feature의 차이
      - Indicator features
        - 가능한 중요한 어떤 것을 가리키기 위해서 data의 다른 part를 사용
  - Feature selection
    - Dimensionality reduction
        - 일반적인 차원 축소 방법.
        - Principal Component Analysis (PCA) 는 더 많은 차원(feature)을 다룰 수 있다.
    - Feature importance (post modelling)
        - 모델을 데이터 셋에 맞춘다.
        - 어떤 feature가 result에 더 중요할 지 조사한다.
        - 가장 중요치 않은 것을 지운다.
    - Wrapper methods (ex genetic algorithm) & recursive feature elimination
        - create large subsets of feature options
        - 중요치 않은 것을 지움
        - 좋은 feature을 찾을 가능성을 높여줌
        - computation time이 많이 들어감
        - TPot이 이걸 함
- Dealing with imbalances
    - 한 클래스엔 10000 개의 데이터가 있고, 다른 한 쪽에는 100 개의 데이터밖에 없는 걸 다루는 문제
        - Collect more data (if you can)
        - scikit-learn-contrib imbalanced-learn package 써보기.
        - SMOTE (synthetic minority over-sampling technique) 쓰기
            - 샘플이 부족한 class의 샘플을 임의로 만들어 줌.
            - scikit-learn-contrib imbalanced-learn package를 이용해서 만들 수 있음.
        - "Learning from Imbalanced Data" paper 참조.
### Data splitting <a id = "data-split"></a>
- Training set (usually 70-80% of data)
    - Model은 이걸 가지고 학습한다.
- Validation set (typically 10-15% of data)
    - 모델 hyperparameter는 이걸 가지고 tune된다.
- Test set (usually 10-15%) 
    - 모델의 최종 성능이 평가되는 곳
    - 옳게만 끝낸다면, 희망적으로 test set에 있는 result는 어떻게 모델이 현실 세계에서 수행하는지에 대해 좋은 indication을 준다.
    - model을 tune하기 위해서 이 dataset을 사용하지 마라.
        
## Train model on data <a id = "train-model"></a>
3 steps: choose an algorithm, overfit the model, reduce overfitting with regularization
### Choosing an algorithm
#### Supervised algorithms
- Linear Regression
    - 그래프에 흩어져 있는 data에 가장 잘 맞는 line을 그리는 것
    - 연속적인 변수를 생산한다 (예를 들어, 인치에 있는 높이)
- Logisitic Regression 
    - 독립적인 변수를 다루는 과정을 바탕으로 두 개의 결과를 예측한다.
    - 병을 갖고 있는지 아닌지를 예측하는 과정
- k-Nearest Neighbours
    - 매우 유사한 'k'개의 예제를 찾는다. 
    - 그러면, 새로운 샘플이 주어졌을 때, 새로운 샘플은 어디와 더 유사한가?
- Support Vector Machines (SVMs)
    - classification 또는 regression에 쓰인다.
    - 많은 평면을 사용해서 data point를 분리하는 가장 좋은 방법을 찾는 것이다. (Hyperplanes라고 불린다.)
- Decision Trees and Random Forests
    - classification과 regression에 쓰인다. (structured data에 매우 좋은 알고리즘이다.)
    - 50이상, 65이하 같은 분류로 데이터를 나눈다.
    - 결론적으로, 더이상 나눌 수 없는 data에서 point를 잡아내는 것이다.(우리가 이걸 정한다.)
    - Random Forest는 많은 decision tree가 결합한 것이다.
    - 효율적으로 많은 모델들을 결합하고, 이점을 얻는다. (이걸 ensembling이라고 부른다.)
    - explained.ai에서 더 훌륭한 방법들을 볼 수 있다.
- AdaBoost/Gradient Boosting Machines (also just known as boosting)
    - Classification과 Regression에 쓰임.
    - 약한 learner를 강한 learner로 바꿀 수 있을까?
    - 종류
        - XGBoost Algorithm
        - CatBoost Algorithm
        - LightGBM Algorithm
    - Neural networks (also called deep learning)
        - Classification 또는 Regression에 사용됨
        - input 여러개를 넣고, linear로 input을 다룸 (weight와 input 사이에서 dot product)
        - nonlinear functions (activation function)
        - 종류
            - Convolutional neural networks (typically used for computer vision)
            - Recurrent neural networks (typically used for sequence modelling)
            - Transformer networks (can be used for vision and text, starting to replace RNNs)

#### Unsupervised algorithms
- Clustering
    - K-Means clustering
        - k개의 클러스터 선택
        - 각각의 클러스터는 랜덤하게 centre node를 받음
        - 각 iteration마다 centre node들은 서로보다 더 멀리 움직임
        - centroid가 움직임을 멈추면, 각각의 샘플은 가까운 centroid에 동등한 가치로 할당됨
- Visualization and dimensionality reduction
    - Principal Component Analysis
        - 많은 차원에서 적은 자원으로 데이터를 줄이는 것.
        - 이럼에도 불구하고, variance를 보존한다.
    - Autoencoders
        - 낮은 차원의 데이터의 encoding을 배운다.
        - 같은 정보를 가지고 있으면서, 100개의 픽셀을 압축해 50개의 픽셀로 나타낼 수 있도록 만드는 기법.
    - t-Distributed Stochastic Neighbor Embedding (t-SNE)
        - 2D 또는 3D 공간에서 높은 차원을 가지고 있는 데이터를 보여주기에 좋다.
- Ananomaly detection
    - Autoencoder
        - 시스템 input들의 차원을 줄이기 위해 사용
        - 어떤 threshold 안에 이러한 input들을 재생성함
        - recreation이 threshold를 맞출 수 없다면, outlier로 분류된다.
    - One-class classification 
        - 오직 하나의 클래스만 학습시키는 모델
        - computer network traffic의 normal event 같은 것
        - class의 바깥쪽에 있으면 이를 anomaly로 부른다.
        - one-class K-Means, one-class SVM, isolation forest, local outlier factor가 여기에 속한다.

### Type of learning<a id = "type-of-learning"></a>
#### Batch learning
- 큰 통계학적 창고에 모든 데이터가 존재한다. 그걸로 모델을 학습시킨다.
- 당신이 새로운 모델을 얻는 날마다 한 달씩 새로운 모델을 돌려야한다.
- 학습하는데 시간이 오래 걸리고, 잘 완료되지 않는다.
- 학습 없이 production에서 실행 (비록 이게 나중에 다시 훈련되어야 하지만)
#### Online learning
- data는 지속적으로 update 된다.
- 지속적으로 새로운 모델에 그걸 학습시킨다.
- 각각의 learning step이 빠르고 쌈
- production에서 실행, 지속적으로 학습
#### Transfer learning
- 하나의 모델이 배운 지식을 취하고 그것을 써라.
- SOTA(state of the art) model 에 이득을 얻는 ability를 줘라.
- data가 많지 않거나, 광범위한 계산 자원이 없다면 유용하다.
- 종류
    - TensorFlow Hub
    - PyTorch Hub
    - Hugging Face transformers (NLP models)
    - Detectron2 (computer vision models)
#### Active learning
- "human in the loop" learning이라고 불린다.
- 인간 전문가가 모델과 상호작용하고, 가장 불확실하게 느끼는 label에 대한 업데이트를 제공하는 것이다.
- 어떻게 Nvidia가 active learning을 사용하는지를 확인해봐라!
#### Ensembling
- 학습의 형태가 아니더라도, 이미 학습된 알고리즘에 더 나은 결과를 내도록 결합하는 알고리즘이다.
- 예를 들면, "wisdom of the crowd" 를 레버리지 하는 것이다.
### Underfitting
- 모델이 수행하지 않을 때 일어난다.
- 더 오래 혹은 더 진보된 모델을 훈련하는 것을 시도한다.
### Overfitting
- validation loss가 증가하기 시작했을 때 발생한다.
- 얼마나 validation dataset에서 잘 수행되는지를 본다, 작을 수록 좋다.
- 또는 validation set을 가지고 있지 않을 때, test set보다 training set에서 얼마나 잘 수행하는 지를 본다. 
- 정규화
    - L1 (lasso) 와 L2(ridge) 정규화:
        - L1 정규화 : 필요하지 않은 특징 변수를 0로 설정
            - 가장 필수적인 곳에 feature selection을 수행하고, 필수적이지 않은 곳에 feature selection을 수행한다.
            - model explainability에 유용하다.
        - L2 정규화 : 0으로 설정하지 않고 model feature를 포함하는 것
        - Dropout : 모델의 부분들을 random하게 지운다. 그래서 나머지 부분들이 더 나아지게 만든다. (일명 가지치기)
        - Early stopping : validation loss가 너무 증가하기 전에(더 일반화 되기 전에) 학습하는 것을 멈추는 것.
            - 다른 어떤 metric도 모델을 향상시키는 것을 멈춘다.
            - Early stopping은 model callback의 형태로 수행된다.
        - Data augmentation : 더 어렵게 배우는 인공적인 방식으로 dataset을 다루는 것.
        - Batch normalization : 다음 레이어로 가기 전 2개의 파라미터(beta, W)를 더하는 것 뿐만 아니라 input을 표준화
            - W : 각각의 layer 마다 parameter를 얼마나 offset 시킬 수 있는지. 그리고 0으로 나누는 것을 피하기 위한 epsilon
            - zero mean과 normalize 방식 이용
            - update할 파라미터가 적기 때문에, 빠른 학습 속도를 가져온다.
            - 일부 network에서는 dropout을 위한 대체가 일어날 수 있다.
### Hyperparameter Tuning <a id = "hyper-tuning"></a>
1. 학습률을 설정한다. (종종 가장 중요한 hyperparameter가 된다.)
    - 일반적으로, 높은 학습률을 가진 알고리즘 -> 새 데이터에 빠르게 적응
    - 최적의 학습률을 찾는다.
        - 낮은 학습률에서부터 시작해 몇 백개의 iteration으로 모델을 학습시킨다.
        - 매우 큰 숫자로 학습률을 천천히 증가시킨다.
        - 학습률과 비교해서 loss를 보여준다(plot) : learning rate를 위해 log scale을 쓴다.
        - U-shaped curve일 경우 최적의 학습법은 밑에서 1-2 notch정도 왼쪽으로 간 것이다.
    - Adam optimizer를 사용하는 학습률 스케쥴링은 학습률을 천천히 낮추는 것을 포함한다. (수렴한다.)
    - Cyclic learning rate: 다이나믹하게 학습을 빠르게 할 수 있는 가능성과 threshold 사이에서 학습률을 올리고 낮춘다.
2. <paper>A disciplined approach to neural network hyperparameters by Leslie Smith.
    - learning rate, batch size, momentum, weight decay and more
3. 조정해야 하는 다른 파라미터
    - layer의 수 (딥러닝을 경우에만)
    - Batch size (모델이 한 번에 얼마나 많은 데이터 예제를 보는 지)
    - tree들의 수 (decision tree algorithm을 했을 때)
    - iteration 수 (얼마나 많은 시간동안 model이 데이터를 통과시킬것이냐)
        - tuning iteration 대신에, early-stopping을 쓴다.

### Analysis/Evaluation <a id = "analysis"></a>
+ Evaluation metrics
    - Classification
        - Accuracy
        - Precision
        - Recall
        - f1
        - Confusion matrix
        - Mean average precision (object detection)
    - Regression
        - MSE (mean squared error)
        - MAE (mean absolute error)
        - R^2 (r-squared)
    - Task-based metric 
        - create one based on your specific problem
+ Feature importance
    - 어떤 특징이 가장 모델에 잘 기여할까?
    - 일부가 지워져야 될까?
    - 모델을 설명할 때 유용함
+ Training/inference time/cost
    - 학습하는 데 얼마나 시간이 오래 걸릴까? 이게 feasible 할까?
    - 추론이 얼마나 걸릴까, 이게 우리의 제품에 적합할까?
+ What-if tool
    - 어떻게 다른 모델과 비교할 수 있을까?
    - 데이터를 더 추가해서 바꾼다면 어떨까?
    - 이게 결과에 얼마나 영향을 줄까?
+ Least confident example
    - 모델이 왜 나빠질까
    - 보통 많은 example을 가질 필요가 없는 긴 tail instance 때문
+ Bias/variance trade-off
    - 높은 bias는 새로운 sample에 대한 일반화의 부족과 underfitting을 일으킨다. 
    - 높은 변수는 랜덤한 noise가 있는 data의 패턴을 찾기 때문에 overfitting을 일으킨다.
    
### Serve model (deploying a model) <a id = "serve-model"></a>
- production으로 모델을 넣고, 어떻게 진행되는 지를 본다.
- vivo에 있는 evaluation metrix들이 매우 훌륭하다. 
- 하지만 진짜로 어떻게 수행되는 지는 알 수가 없다.
- Tools you can use
    - TensorFlow Serving (part of TFX, TensorFlow Extended)
    - PyTorch Serving (TorchServe)
    - Google AI Platform : REST API로 너의 모델을 가능하게 만들어라
    - Sagemaker
- MLOps
    - 소프트웨어 엔지니어링이 머신 러닝을 만족시키는 곳
    - 머신 러닝 모델 주위에서 필요로 하는 모든 기술들이 production 동안에 일하도록 하는 것을 필요로 한다.
    
### Retrain model <a id = "retrain-model"></a>
- 다양한 evaluation metric들에 기초하여 serving 후 모델이 어떻게 수행하는 지를 보는 것 (또는 serving 이전에)
- 필요한 만큼 위에 있는 step을 반복 (machine learning은 experimental하다, 그러므로 여기가 당신의 data와 experiment가 추적하기를 원하는 곳임을 기억하라)
- data source가 변하거나 upgrade될 때 'age'(fine-wine style에 있지 않은)나 'drift'에서 시작하는 모델 예측을 찾을 수 있다.
- 이건 재학습을 원할 때 하면 된다.
