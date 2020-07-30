# 알아보기 쉽게 정리한 머신러닝 roadmap

## machine learning data processing

### Data Types
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

|    |Normal|Ordinal|Interval|Ratio
|----|----|----|----|----
Counts / Distribution| O | O | O | O 
Minimum, Maximum|    | O | O | O 

### Data Exploration

### Feature Cleaning

### Feature Imputation

### Feature Engineering

### Feature Selection

### Feature Encoding

### Feature Normalisation or Scaling

### Dataset Construction 



출처 :
https://github.com/dformoso/machine-learning-mindmap 
https://www.graphpad.com/support/faq/what-is-the-difference-between-ordinal-interval-and-ratio-variables-why-should-i-care/
