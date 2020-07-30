# machine learning data processing

## Data Types
데이터 
**Nominal**<br>
is for mutual exclusive, but not ordered, categories<br>
= 서로가 서로에게 어떠한 영향도 미치지 않는 것, 순서, 카테고리 x<br>

> ex : genotype, blood type, zip code, gender, race, eye color, political party<br>

**Ordinal**<br>
is one where the order matters but not the difference between values<br>
= 순서는 중요하지만 value 사이에 차이 x<br>

> ex : socio economic status (“low income”,”middle income”,”high income”)<br>
> education level (“high school”,”BS”,”MS”,”PhD”), <br>
> income level (“less than 50K”, “50K-100K”, “over 100K”), <br>
> satisfaction rating (“extremely dislike”, “dislike”, “neutral”, “like”, “extremely like”).

**Interval** <br>
is a measurement where the difference between two values is meaningful <br>
= 두 개의 value의 차이가 의미있는 값 <br>

> ex : emperature (Farenheit), temperature (Celcius), pH, SAT score (200-800), credit score (300-850).

**Ratio**
has all the properties of an interval variable, and also has a clear definition of 0.0. <br>
interval variable 의 모든 property를 가지고 있고, 0.0.의 명확한 정의 <br>

> enzyme activity, dose amount, reaction rate, flow rate, concentration, pulse, weight, length, temperature in Kelvin (0.0 Kelvin really does mean “no heat”), survival time.

|    |Norminal|Ordinal|Interval|Ratio
|:----|:----:|:----:|:----:|:----:
Counts / Distribution| O | O | O | O 
Minimum, Maximum|    | O | O | O |   
Range|    | O | O | O |
Percentiles|    | O | O | O |
Standard deviation, Variable|  |   | O | O | 

|    | Nominal | Ordinal| Interval | Ratio 
|:----|:----:|:----:|:----:|:----:
Mode|O|O|O|O
Median||O|O|O
Mean|||O|O

|    | Nominal | Ordinal| Interval | Ratio 
|:----|:----:|:----:|:----:|:----:
Countable|O|O|O|O
Order defined||O|O|O
Difference defined (addition, subtraction) |||O|O
Zero defined (multiplication, division) ||||O

## Data Exploration
데이터 
### Variable Identification
Identify Predictor (Input) and Target(output) variables<br>
Next, identify the data type and category of the variables<br>
= Input, Output 정의 -> data type과 변수 카테고리 정의 <br>

### Univariate Analysis
* Continuous Features (연속된 특징)
Mean, Median, Mode, Min, Max, Range, Quartile, IQR, Variance, Standard, Deviation, Skewness, Histogram, Box Plot

* Categorical Features (분류된 특징)
Frequency, Histogram

### Bi-variate Analysis
Finds out the relationship between two variables<br>
= 두개의 variable 사이에서 관계를 찾음

#### Numerical & Numerical <br>
**Scatter Plot**<br>
is usually drawn before working out a linear correlation or fitting a regression line<br>
= linear correlation을 하기 전에 그리거나, regression line을 맞추기 위함<br>
<br>
**Correlation Plot** - Heatmap<br>
quantifies the strength of a linear relationship between two numerical variables<br>
= 2개의 numerical variable 사이에서 linear relationship의 강점을 수량화한다.<br><br>

#### Categorical & Categorical
**Two-way table(=contigency table)**<br>
can start analyzing the relationship by creating a two way table of count and count%<br>
= count와 count%에 대한 두가지 방법의 table을 생성함으로서 관계를 분석하는 것<br><br>

**Stacked Column Chart**<br>
compares the percentage that each category from one variable contributes to a total across categories of the second variables<br>
= 하나의 variable에서 나온 각각의 카테고리가 2번째 variable의 카테고리를 가로질러 전체에게 영향을 주는 비율.<br><br>

**Combination Chart** <br>
two or more chart types to emphasize that the chart contains different kinds of information<br><br>

**Chi-Square Test** <br>
can be used to determine the association between categorical variables<br>
This test is used to derive the statistical significance of relationship between the variables<br><br>

#### Categorical & Numerical
Line Chart with Error Bars
Z-Test / T-Test<br>
ANOVA<br>

## Feature Cleaning 
특징을 명확하게
### Missing values
<br>
One may choose to either omit elements from a dataset that contain missing values or to impute a value
= missing value를 포함하고 있는 dataset에서 수정할 element를 선택 하거나 value를 삽입하는 것<br><br>

### Special Values
<br>
Numeric variables are endowed with several formalized special values including +-Inf, NA and NAN.<br>
Calculations involving special values often result in special values, and need to be handled/cleaned <br>
= Numerica variable들은 +-Inf, Na, NAN을 포함한 특별한 value로 여러번 formalize될 수 있다.
= 특별한 value들을 포함한 계산은 특별한 value로 나온다. 그리고 다뤄지거나 삭제될 필요가 있다.

### Outliers
<br>
They should be detected, but not necessarily removed.
Their inclusion in the analysis is a statistical decision.
이들은 발견될 수는 있지만, 필수적으로 삭제되지는 않는다.
분석에서 이들의 포함은 통계적인 결정이다.

### Obvious inconsistencies
A person's age cannot be negative, a man cannot be pregnant and under-aged person cannot possess a drivers license.
= 절대로 될 수 없는 명제 ex) 사람의 나이가 음수

## Feature Imputation
특징 삽입
### Hot-Deck
The technique then finds the first missing value and uses the cell value immediately prior to the data that are missing to impute the missing value<br>
= first missing value를 찾음, missing value를 imput하는 것을 잃어버리는 데이터 이전에 cell value를 즉각적으로 찾음. <br><br>

### Cold-Deck
Selects donors from another dataset to complete missing data<br>
= missing data를 완성시키기 위해서 다른 dataset에서 donor를 선택<br><br>

### Mean-substitution
Another imputation technique involves replacing any missing value with the mean of that variable for all other cases, which has the benefit of not changing the sample mean for that variable.<br>
= 다른 삽입 기술은 그 varibale을 위한 샘플 평균을 변화시키는 것이 아닌 것으로 산출되는 이득을 가지는 모든 cases를 위해 있는 그 variable 의 평균으로 어떤 missing value를 대체하는 것을 포함한다.<br><br>

### Regression
A regression model is estimated to predict observed values of a variable based on other variables, and that model is then used to impute values in cases where that variable is missing <br>
= 다른 variable에 기반한 variable의 observed value를 예측하도록 추정되는 모델.<br>
= 이 모델은 variable이 사라진 곳에 value를 넣는데 사용한다.<br><br>

## Feature Engineering

### Decompose
Converting 2014-09-20T20:45:40Z into categorical attributes like hour_of_the_day, part_of_day, etc. <br>
= 2014-09-20T20:45:40Z를 카테고리 속성으로 바꿈 : hour_of_the_day, part_of_day 같은 식으로<br><br>

### Discretization
**Continuous Features**<br>
Typically data is discretized into partitions of K equal lengths/width (equal intervals) or K% of the total data (equal frequencies)<br>
= 일반적으로 data는 K개의 같은 길이/넓이 (같은 구간)의 조각 혹은 모든 데이터 (주파수가 동등한)의 K%로 나눠진다.<br><br>

**Categorical Features**<br>
Values for categorical features may be combined, particularly when there’s few samples for some categories<br>
= 특히 어떤 카테고리에 매우 적은량의 sample만 있을 때 카테고리로 만들 수있는 특징들은 결합된다. <br><br>

### Reframe Numerical Quantities
Changing from grams to kg, and losing detail might be both wanted and efficient for calculation<br>
= gram을 kg으로 바꾸면서, detail을 잃는 것은 원해서 그러거나 계산을 효율적으로 하기 위함이다.<br><br>

### Crossing

Creating new features as a combination of existing features. <br>
Could be multiplying numerical features, or combining categorical variables.<br> 
This is a great way to add domain expertise knowledge to the dataset.<br><br>

= 현재 있는 특징들을 결합함으로서 새로운 특징들을 만든다.<br>
= 셀 수 있는 특징들을 곱하거나 카테고리 변수들을 결합할 수 있다 <br>
= 데이터 셋에 전문적인 지식의 domain을 추가하는 아주 좋은 방법이다. <br><br>

## Feature Selection
### Correlation
Features should be uncorrelated with each other and highly correlated to the feature we’re trying to predict.
[correlation 1]<br><br>
**Convariance** :  A measure of how much two random variables change together. <br> 
Math: dot(de_mean(x), de_mean(y)) / (n - 1)<br><br>

### Dimensionality Reduction
**Principal Component Analysis(PCA)**
Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.<br><br>

This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components.<br><br>

= PCA는 가능한 상호 연관된 변수(possibly correlated variables)들의 관찰 집합을 principal component라고 불리는 선형 비상호 연관 변수(linearly uncorrelated variables)들의 값의 집합으로 변환하는 직교 변환을 사용하는 통계적인 과정이다.<br>
이런 변형은 first principal component가 가장 큰 가능한 변수를 갖는것이다. <br>
(이 말은, data에서 가능한 한 variability를 많이 설명한다.) <br>
그리고, 각각의 연속적인 component는 앞서 가는 component들에게 직교해야 한다는 제한 아래에서 차례로 가능한 가장 높은 variance를 가진다.<br>
<br>
Plot the variance per feature and select the features with the largest variance.<br><br>
= feature별로 변수를 plot해라, 그리고 가장 큰 변수로 feature를 골라라.<br><br>

**Singular Value Decomposition(SVD)**
SVD is a factorization of a real or complex matrix.<br> 
It is the generalization of the eigendecomposition of a positive semidefinite normal matrix (for example, a symmetric matrix with positive eigenvalues) to any m×n matrix via an extension of the polar decomposition.<br> 
It has many useful applications in signal processing and statistics.<br><br>

= SVD는 real or complex 행렬의 인수분해이다.<br>
= 양극 분해(polar decomposition)의 확장을 통해 모든 mxn 행렬에 대한 positive semidefinite normal matrix(0이 포함되고, 모든 행렬이 양수인 정규행렬 예를 들어, positive 고유벡터를 가지고 있는 sysmmetric matrix 다) 의 고유값분해(eigendecomposition)의 일반화이다.<br>
= 시그널 프로세싱과 통계학에 많이 적용된다. <br><br>


### Importance
#### Filter Methods 
Filter type methods select features based only on general metrics like the correlation with the variable to predict.<br> 
Filter methods suppress the least interesting variables. <br>
The other variables will be part of a classification or a regression model used to classify or to predict data.<br>
These methods are particularly effective in computation time and robust to overfitting.<br><br>

Filter type method는 예측하기 위한 variable으로 하는 correlation 처럼 일반적인 행렬에 기본을 두고 특징을 선택한다. <br>
Filter method는 덜 흥미로운 변수들을 압도한다.<br>
다른 variable들은 classification의 부분이거나 데이터를 분류하거나 예측하기 위해 사용되는 regression model의 부분이 된다.<br>
이러한 방법은 계산 시간에서 특히 효과적이다. 그리고 overfitting에 robust하다.<br><br>
* Correlation
* Linear Discriminant Analysis
* ANOVA: Analysis of Variance
* Chi Square

#### Wrapper Methods
Wrapper methods evaluate subsets of variables which allows, unlike filter approaches, to detect the possible interactions
between variables. <br>
The two main disadvantages of these methods are : The increasing overfitting risk when the number of observations is insufficient.<br> 
AND. The significant computation time when the number of variables is large.<br><br>

Wrapper methods는 filter 접근과는 다르게 variable 사이에서 가능한 상호작용을 탐지하는 것을 허락하는 variable의 subset을 평가하는 것이다.<br>
이러한 방법들에서 안 좋은 2가지 점이 있다.<br>
첫 번째는, observation의 수가 불충분할 때 overfitting risk가 증가<br>
두 번째는, variable의 수가 많을 때 중요한 계산 시간이다.<br><br>

* Forward Selection
* Backward Elimination


#### Embedded Methods
Embedded methods try to combine the advantages of both previous methods.<br>
A learning algorithm takes advantage of its own variable selection process and performs feature selection and classification simultaneously.<br><br>


## Feature Encoding

## Feature Normalisation or Scaling

## Dataset Construction 



[correlation1] 

출처 :
https://github.com/dformoso/machine-learning-mindmap 
https://www.graphpad.com/support/faq/what-is-the-difference-between-ordinal-interval-and-ratio-variables-why-should-i-care/
https://www.saedsayad.com/bivariate_analysis.htm

