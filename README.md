Implementation of a simple MLP from scratch in C. It does not use any external libraries.
It has limited features as I want to keep it simple and easy to learn from.

[Micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy was a big help.

## Build and Run (Windows)
To build the train.c file with gcc:
```
$  gcc -o train train.c src/*.c
```
then run with:
```
$  train.exe
```

If you want to run the tests, use the following code and replace tests/test.c with the test you want e.g.
```
$  gcc -o test_tensor_ops tests/test_tensor_ops.c src/*.c
```

## Demo
The train.c file contains the training loop for a binary classifier with two hidden layers of size 16. Binary cross entropy loss is used with SGD as the optimizer. The dataset is the [moons dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html). This setup is identical to the demo from the previously mentioned micrograd so that I can compare performance; however, I used binary cross entropy instead of hinge loss.
Here is an example decision boundary after 100 iterations using 100 data samples:

![demo decision boundary after 100 iterations](decision_boundary.png)

## Visualising
The visualise_decision_boundary.ipynb file will display the decision boundary of the binary classifier by reading data that train.c exports. Train.c will pass points on a grid through the trained model and export the points along with the models prediction.
Ensure that "n_x_steps" and "n_y_steps" in both train.c and visualise_decision_boundary.ipynb match.
Matplotlib and Numpy must be installed.
