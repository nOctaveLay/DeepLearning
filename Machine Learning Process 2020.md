
# Index

### Data collection
### Data preparation
### Train model on data
1. choose an algorithm
1. overfit the model
1. reduce overfitting with regularization
### Analysis/Evaluation
### Serve model (deploying a model)
### Retrain model

## Data Collection
### Question to ask
* What kind of problem are we trying to solve? (see machine learning problems)
* What data sources already exist?
* What privacy concerns are there?
* Is the data public?
* Where should we store the data?

### Types of data
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

## Data Preparation
### Exploratory data analysis (EDA), learning about the data you're working with
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
### Data preprocessing, preparing your data to be modelled
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
### Data splitting
- Training set (usually 70-80% of data)
    - Model은 이걸 가지고 학습한다.
- Validation set (typically 10-15% of data)
    - 모델 hyperparameter는 이걸 가지고 tune된다.
- Test set (usually 10-15%) 
    - 모델의 최종 성능이 평가되는 곳
    - 옳게만 끝낸다면, 희망적으로 test set에 있는 result는 어떻게 모델이 현실 세계에서 수행하는지에 대해 좋은 indication을 준다.
    - model을 tune하기 위해서 이 dataset을 사용하지 마라.
        
## Train model on data
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
    - 
#### Unsupervised algorithms

### Type of learning
### Underfitting
### Overfitting
### Hyperparameter Tuning 

        
