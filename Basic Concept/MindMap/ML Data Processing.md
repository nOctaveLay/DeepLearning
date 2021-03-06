# Machine Learning Data Processing

## Index

- [Data Types](#Data%20Types)
- [Data Exploration](#Data%20Exploration)
  - [Variable Identification](#Variable%20Identification)
  - [Univariate Analysis](#Univariate%20Analysis)
  - [Bi-variate Analysis](#Bi-variate%20Analysis)
  - [Numerical & Numerical](#Numerical%20&%20Numerical)
  - [Categorical & Categorical](#Categorical%20&%20Categorical)
  - [Categorical & Numerical](#Categorical%20&%20Numerical)
- [Feature Cleaning](#Feature%20Cleaning)
  - [Missing values](#Missing%20values)
  - [Special Values](#Special%20Values)
  - [Outliers](#Outliers)
  - [Obvious inconsistencies](#Obvious%20inconsistencies)
- [Feature Imputation](#Feature%20Imputation)
  - [Hot-Deck](#Hot-Deck)
  - [Cold-Deck](#Cold-Deck)
  - [Mean-substitution](#Mean-substitution)
  - [Regression](#Regression)
- [Feature Engineering](#Feature%20Engineering)
  - [Decompose](#Decompose)
  - [Discretization](#Discretization)
  - [Reframe Numerical Quantities](#Reframe%20Numerical%20Quantities)
  - [Crossing](#Crossing)
- [Feature Selection](#Feature%20Selection)
  - [Correlation](#Correlation)
  - [Dimensionality Reduction](#Dimensionality%20Reduction)
  - [Importance](#Importance)
    - [Filter Methods](#Filter%20Methods)
    - [Wrapper Methods](#Wrapper%20Methods)
    - [Embedded Methods](#Embedded%20Methods)
    - [Feature Encoding](#Feature%20Encoding)
    - [Feature Normalisation or Scaling](#Feature%20Normalisation%20or%20Scaling)
- [Dataset Construction](#Dataset%20Construction)
  - [Training Dataset](#Training%20Dataset)
  - [Test Dataset](#Test%20Dataset)
  - [Validation Dataset](#Validation%20Dataset)
  - [Cross Validation](#Cross%20Validation)

## Data Types

Nominal

>is for mutual exclusive, but not ordered, categories
>
>= 서로가 서로에게 어떠한 영향도 미치지 않는 것, 순서, 카테고리 x
>
> ex : genotype, blood type, zip code, gender, race, eye color, political party

Ordinal

> is one where the order matters but not the difference between values
>
> = 순서는 중요하지만 value 사이에 차이 x
>
> ex : socio economic status (“low income”,”middle income”,”high income”),
> education level (“high school”,”BS”,”MS”,”PhD”),
> income level (“less than 50K”, “50K-100K”, “over 100K”),
> satisfaction rating (“extremely dislike”, “dislike”, “neutral”, “like”, “extremely like”).

Interval

>is a measurement where the difference between two values is meaningful
>
>= 두 개의 value의 차이가 의미있는 값
> ex : emperature (Farenheit), temperature (Celcius), pH, SAT score (200-800), credit score (300-850).

Ratio

>has all the properties of an interval variable, and also has a clear definition of 0.0.
>
>= interval variable 의 모든 property를 가지고 있고, 0.0.의 명확한 정의
>
> ex : enzyme activity, dose amount, reaction rate, flow rate, concentration, pulse, weight, length, temperature in Kelvin (0.0 Kelvin really does mean “no heat”), survival time.

|    | Nominal | Ordinal| Interval | Ratio
|:----|:----:|:----:|:----:|:----:
Mode|O|O|O|O
Median||O|O|O
Mean|||O|O

|    |Norminal|Ordinal|Interval|Ratio
|:----|:----:|:----:|:----:|:----:
Counts / Distribution| O | O | O | O
Minimum, Maximum|    | O | O | O |
Range|    | O | O | O |
Percentiles|    | O | O | O |
Standard deviation, Variable|  |   | O | O |

|    | Nominal | Ordinal| Interval | Ratio
|:----|:----:|:----:|:----:|:----:
Countable|O|O|O|O
Order defined||O|O|O
Difference defined (addition, subtraction) |||O|O
Zero defined (multiplication, division) ||||O

## Data Exploration

> 특징 탐색

### Variable Identification

- Predictor (Input) 과 Target(output) 변수들을 정의
- 그 다음, 변수들의 Data type과 카테고리 정의

### Univariate Analysis

- Continuous Features (연속된 특징)
  - Mean, Median, Mode, Min, Max, Range, Quartile, IQR, Variance, Standard, Deviation, Skewness, Histogram, Box Plot

- Categorical Features (분류된 특징)
  - Frequency, Histogram

### Bi-variate Analysis

- 두개의 variable 사이에서 관계를 찾음

#### Numerical & Numerical

- Scatter Plot
  - linear correlation을 하기 전에 혹은 regression line을 맞추기 전에 그리는 것
- Correlation Plot - Heatmap
  - 2개의 numerical variable 사이에서 linear relationship의 강점을 수량화한다.

#### Categorical & Categorical

- Two-way table(=contigency table)
  - count와 count%에 대한 두가지 방법의 table을 생성함으로서 관계를 분석하는 것을 시작할 수 있다.
- Stacked Column Chart
  - 하나의 variable에서 나온 각각의 카테고리가 2번째 variable의 카테고리 전체에게 영향을 주는 비율을 비교
- Combination Chart
  - 차트가 다른 종류의 정보를 포함하고 있는 것을 강조하기 위한 2개 이상의 chart type
- Chi-Square Test
  - categorical variable들 사이에서의 결합을 결정하기 위해 사용된다.
  - 이 test는 변수들 사이에서 관계의 통계학적인 중요성을 얻기 위해서 사용된다.

#### Categorical & Numerical

- Line Chart with Error Bars
- Z-Test / T-Test
- ANOVA

## Feature Cleaning

> 특징 제거

### Missing values

- 어떤 사람은 missing value를 포함하고 있는 dataset에서 element를 수정하는 것을 선택하거나 value를 삽입하는 것을 선택한다.

### Special Values

- Numerica variable들은 +-Inf, Na, Nan같은 여러번 formal된 특별한 value들이 부여되었다.
- 특별한 value들을 포함한 계산들은 종종 특별한 value로 나온다. 그리고 이러한 변수들은 다뤄지거나 삭제될 필요성이 있다.

### Outliers

- 이들은 발견될 수는 있지만, 필수적으로 삭제되지는 않는다.
- 분석에서 이들을 포함하는 것은 통계에 의한 결정이다.

### Obvious inconsistencies

- 절대로 현실에서 있을 수 없는, 당연한 명제
- 사람의 나이는 음수가 될 수 없고, 남자는 임신을 할 수 없으며, 어린 아이들은 운전 면허를 가질 수 없다.

## Feature Imputation

> 특징 삽입

### Hot-Deck

- 처음에 missing value를 찾는다.
- Missing value를 삽입하는 것을 잊어버리는 데이터를 사용하기 전에 cell value를 즉시 사용한다.

### Cold-Deck

- Missing data를 완성시키기 위해서 다른 dataset에서 donor를 선택

### Mean-substitution

- 다른 삽입 기술은 어떤 missing value를 모든 다른 경우에 대한 variable의 평균으로 대체하는 것을 포함한다.
- 이는 variable을 위한 sample mean을 바꾸지 않아도 된다라는 장점을 가지고 있다.

### Regression

- 이 모델은 다른 variable들을 기준으로 variable의 관측값을 예측하는 것으로 추정된다.
- 그리고 이 모델은 variable이 사라진 경우 value를 넣기 위해서 사용된다.

## Feature Engineering

### Decompose

- 2014-09-20T20:45:40Z를 카테고리 속성으로 바꿈
  - hour_of_the_day, part_of_day 등

### Discretization

- Continuous Features
  - 일반적으로 data는 K개의 같은 길이/넓이 (같은 구간)의 조각 혹은 모든 데이터 (주파수가 동등한)의 K%로 나눠진다.
- Categorical Features
  - 특히 어떤 카테고리에 매우 적은량의 sample만 있을 때 카테고리로 만들 수있는 value들은 결합된다.

### Reframe Numerical Quantities

- gram을 kg으로 바꾸는 것이나 detail을 잃는 것은 원해서 그러거나 계산에 있어서 효율적이기 때문이다.

### Crossing

- 현재 있는 특징들을 결합함으로서 새로운 특징들을 만든다.
- 셀 수 있는 특징들을 곱하거나 카테고리 변수들을 결합할 수 있다.
- 데이터 셋에 전문적인 지식의 domain을 추가하는 아주 좋은 방법이다.

## Feature Selection

### Correlation

- 특징들은 서로 연관이 없어야 하며, 우리가 예측하려고 하는 특징들과 많이 연관 되어 있어야한다.
![correlation 1](./images/correlation.PNG)
- **Convariance** :  얼마나 두 개의 random variable들이 서로 서로 변화할 수 있는 지에 대한 척도
- Math: dot(de_mean(x), de_mean(y)) / (n - 1)

### Dimensionality Reduction

- Principal Component Analysis(PCA)
  - PCA는 가능한 상호 연관된 변수(possibly correlated variables)들의 관찰 집합을 principal component라고 불리는 선형 비상호 연관 변수(linearly uncorrelated variables)들의 값의 집합으로 변환하는 직교 변환을 사용하는 통계적인 과정이다.
  - 이러한 transformation은 첫번째 principal component가 가장 큰 값을 가지는 variance를 갖는 것으로 정의된다.
  - 이 말은 즉슨, 데이터 안에서 가능한 한 많은 variability를 갖는 것을 설명한다.
  - 그리고 각각의 연속적인 component는 차례로 앞에 있는 component들과 직교해야 한다는 조건 아래 가능한 가장 큰 variance를 순서대로 갖는다.
  - feature별로 변수를 plot해라, 그리고 가장 큰 variance를 가진 feature를 골라라.

- Singular Value Decomposition(SVD)
  - SVD는 실수 혹은 복소수 행렬의 인수분해이다.
  - 이것은 어떤 mxn 행렬에 대해 양극 분해의 확장을 통하여 정부호행렬(positive semi-define normal matrix, 예를 들면 양수의 고유값을 가지고 있는 symmetric 행렬)의 고유값분해로 일반화할 수 있다.
  - signal processing과 통계학에서 유용하게 많이 사용되고 있다.

### Importance

#### Filter Methods

- Filter type method는 예측하기 위해 사용되는 variable을 가지고 하는 correlation 처럼 일반적인 행렬들에 기반하여 특징을 선택한다.
- Filter method는 가장 흥미롭지 않는 변수들을 압도한다.
- 다른 variable들은 classification의 일부거나 데이터를 분류하거나 예측하기 위해 사용되는 regression model의 일부가 된다.
- 이러한 방법은 계산 시간에서 특히 효과적이다. 그리고 overfitting에 robust하다.
- 종류
  - Correlation
  - Linear Discriminant Analysis
  - ANOVA: Analysis of Variance
  - Chi Square

#### Wrapper Methods

- Wrapper methods는 filter 접근과는 다르게 variable 사이에서 가능한 상호작용을 탐지하는 것을 허락하는 variable의 부분 집합을 평가하는 것이다.
- 2가지 단점이 있다.
  - Observation의 수가 불충분할 때 증가하는 overfitting risk
  - Variable의 수가 많을 때 생기는 어마어마한 계산 시간
- 종류
  - Forward Selection
  - Backward Elimination
  - Recursive Feature Ellimination
  - Genetic Algorithms

#### Embedded Methods

- Embedded method는 과거의 method들의 장점들을 결합하는 것을 시도한다.
학습 알고리즘은 알고리즘의 변수 선택 과정에서 이점을 얻는다.
- 특징 선택과 분류를 동시에 한다.
- 예시
  - Lasso regression: coefficient의 크기의 **절댓값**과 동일하게 페널티를 더하는 L1 정규화를 한다.
  - Ridge regression은 coefficient의 크기의 **제곱**과 동등하게 페널티를 더하는 L2 정규화를 한다.

## Feature Encoding

- 머신 러닝 알고리즘은 선형 대수를 적용한다.
- 이 말은 모든 feature들은 반드시 numeric 해야 함을 의미한다.
- Encoding은 모든 feature들이 numeric하도록 돕는다.
  - Label Encoding
  - One Hot Encoding
    - One Hot Encoding에서 반드시 encoding이 모든 feature들이 선형 독립적이어야 한다는 것을 보장해라

## Feature Normalisation or Scaling

- objective function(목적 함수)은 normalization(정규화)없이는 작동 x
  - raw data의 값의 범위가 넓기 때문
- Feature scaling이 적용되는 이유
  - gradient descent가 feature scaling과 함께 하면 더 빠르게 수렴되기 때문.
- Method
  - Rescaling
    - 가장 단순한 방법: [0,1] [-1,1] 범위 안으로 들이기 위해서 feature의 범위를 리스케일링하는 것
  - Standardization
    - Feature standardization은 data안에서 각각의 특징들에 대한 value들이 zero-mean (numerator에서 평균을 빼는 것)과 unit-variance를 가지게 만드는 것이다.
  - Scaling to unit length
    - Feature vector의 컴포넌트를 증가시키려 한다.
    - 그래서 complete vector는 length one을 가진다.

## Dataset Construction

### Training Dataset

- 학습을 위해 사용되는 예제.
- 예를 들어, Multilayer Perceptron에서 classifier의 parameter를 맞추기 위해, back propagation을 할 때 **최적의** weight를 찾기 위해서 training set을 사용한다.

### Test Dataset

- 완전히 훈련된 classifier의 performance을 측정할 때 사용되는 예제.
- Multilayer Perceptron의 경우, 최종 모델 (MLP size 그리고 실제 weight들)을 고른 후에 error rate를 측정하기 위해서 test set을 사용한다.
- test set에서 최종 모델을 평가한 후, 더 이상 모델을 조정하면 안된다.

### Validation Dataset

- classifier의 parameter를 조정하는데 사용되는 예제.
- Multilayer Perceptron의 경우, hidden unit의 최적의 갯수를 찾고 싶거나, back-propagation algorithm에서 stopping point를 결정하기 위해 validation set을 사용한다.

### Cross Validation

- cross-validation의 첫번째 라운드는 하나의 subset에 대한 분석을 수행하는 동안 데이터 샘플을 차집합으로 분리하는 것을 말한다. (이를 training set이라고 부른다.)
- 그리고 다른 subset에 대한 유효성 분석을 한다. (이를 validation set 혹은 testing set이라고 부른다)
- variability를 줄이기 위해서, cross-validation이 실행되는 여러 round는 다른 partition들을 사용해서 수행된다.
- validation result는 round마다 평균이 난다.

## 출처

- [Mindmap](https://github.com/dformoso/machine-learning-mindmap)
- [Additional Article of Data Type](https://www.graphpad.com/support/faq/what-is-the-difference-between-ordinal-interval-and-ratio-variables-why-should-i-care/)
- [Additional Article of Bivariate Analysis](https://www.saedsayad.com/bivariate_analysis.htm)
