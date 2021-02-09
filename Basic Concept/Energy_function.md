# What is Energy Function?

- Energy function is minimized by inference process
- the goal is to find such value of ```Y```, such that ```E(X,Y)``` takes is minimal values.

## Diffierence of loss function and Energy function

- Loss Function is a measure of a quality of an energy function using training set
- the energy function describes our problem
- the loss function is just something that is used by an ML algorithm as input
- this might be the same function but is not necessarily the case.

> The energy of a system in physics might be the movement inside this system. In a ML context, you might want to minimize the movement by adjusting the parameters. Then one way to achieve this is to use the energy function as a loss function and minimize this function directly. In other cases this function might not be easy to evaluate or to differentiate and then other functions might be used as a loss for your ML algorithm. Similarly as in classification, where you care for the accuracy of the classifier, but you still use cross entropy on the softmax as a loss function and not accuracy.


## 출처

[Energy function](https://stackoverflow.com/questions/50342526/what-is-the-difference-between-energy-function-and-loss-function)
