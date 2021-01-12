# 기초지식 -1 Softmax function

+ 다른 말로 normalizaed exponential function이라고도 불린다.
+ 통계학에서 Boltzmann Distribution (또는 Gibbs distribution)이라고 알려져있다.
+ K 개의 실수의 z 벡터를 input으로 받고, 이걸 K probailities proportional을 input number의 exponential로 바꾸는 과정을 포함하면서 probability distribution으로 바꾼다.
  - softmax를 적용하기 전에 반드시 벡터의 원소가 음수거나, 1보다 커야되고 총합이 1이 아니어야 됨을 의미한다.
  - 그렇지만 softmax를 적용하면 반드시 구간 (0,1)에 있게 된다
  - component들은 1까지 더해진다. 그러므로 확률로 해석할 수도 있다.
  - 높은 input component들은 높은 확률을 가진다.
  - 
출처 : https://en.wikipedia.org/wiki/Softmax_function
