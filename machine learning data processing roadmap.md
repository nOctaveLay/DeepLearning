# machine learning data processing

## Data Types
**Nominal**<br>
is for mutual exclusive, but not ordered, categories<br>
서로가 서로에게 어떠한 영향도 미치지 않는 것, 순서, 카테고리 x<br>

> ex : genotype, blood type, zip code, gender, race, eye color, political party<br>

**Ordinal**<br>
is one where the order matters but not the difference between values<br>
순서는 중요하지만 value 사이에 차이 x<br>

> ex : socio economic status (“low income”,”middle income”,”high income”)<br>
> education level (“high school”,”BS”,”MS”,”PhD”), <br>
> income level (“less than 50K”, “50K-100K”, “over 100K”), <br>
> satisfaction rating (“extremely dislike”, “dislike”, “neutral”, “like”, “extremely like”).

**Interval** <br>
is a measurement where the difference between two values is meaningful <br>
두 개의 value의 차이가 의미있는 값 <br>

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
### Variable Identification
Identify Predictor (Input) and Target(output) variables<br>
Next, identify the data type and category of the variables<br>
Input, Output 정의 -> data type과 변수 카테고리 정의 <br>

### Univariate Analysis
* Continuous Features (연속된 특징)
Mean, Median, Mode, Min, Max, Range, Quartile, IQR, Variance, Standard, Deviation, Skewness, Histogram, Box Plot

* Categorical Features (분류된 특징)
Frequency, Histogram

### Bi-variate Analysis
Finds out the relationship between two variables<br>
두개의 variable 사이에서 관계를 찾음

#### Numerical & Numerical <br>
**Scatter Plot**<br>
is usually drawn before working out a linear correlation or fitting a regression line<br>
linear correlation을 하기 전에 그리거나, regression line을 맞추기 위함<br>
<br>
**Correlation Plot** - Heatmap<br>
quantifies the strength of a linear relationship between two numerical variables<br>
2개의 numerical variable 사이에서 linear relationship의 강점을 수량화한다.<br><br>

#### Categorical & Categorical
**Two-way table(=contigency table)**<br>
can start analyzing the relationship by creating a two way table of count and count%<br>
count와 count%에 대한 두가지 방법의 table을 생성함으로서 관계를 분석하는 것<br><br>

**Stacked Column Chart**<br>
compares the percentage that each category from one variable contributes to a total across categories of the second variables<br>
하나의 variable에서 나온 각각의 카테고리가 2번째 variable의 카테고리를 가로질러 전체에게 영향을 주는 비율.<br><br>

**Combination Chart** <br>
two or more chart types to emphasize that the chart contains different kinds of information<br><br>

**Chi-Square Test** <br>
can be used to determine the association between categorical variables<br>
This test is used to derive the statistical significance of relationship between the variables<br><br>

#### Categorical & Numerical
Z-Test / T-Test
ANOVA

## Feature Cleaning

## Feature Imputation

## Feature Engineering

## Feature Selection

## Feature Encoding

## Feature Normalisation or Scaling

## Dataset Construction 



출처 :
https://github.com/dformoso/machine-learning-mindmap 
https://www.graphpad.com/support/faq/what-is-the-difference-between-ordinal-interval-and-ratio-variables-why-should-i-care/
https://www.saedsayad.com/bivariate_analysis.htm

