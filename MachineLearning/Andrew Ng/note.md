# ML note of Andrew Ng's course

[TOC]

## W1

### 1. Fundamentals

- **Supervised Learning**

Regression (prediction)

Classification

...

- **Unsupervised Learning**

Clustering (grouping)

Non-clustering

...

### 2. Model Representation

***Glossary #1***

$m$:  the size of training set

$x^{(i)}$ : input of the training set

$y^{(i)}$: output of the training set

**Linear Regression Problem:**

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220225203053421.png" alt="image-20220225203053421" style="zoom:50%;" />

where $h_{\theta}(x)=\theta_{0}+\theta_{1} x$

### 3. Cost Function

#### a. definition

Training dataset input $x$ and output $y$ known, we need to find $\theta_0$ and $\theta_1$ s.t.  

$$
J(\theta_0,\theta_1)=\frac{1}{2m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$

 get the minimum value

formally, we are going to find

$$
\underset{\theta_{0}, \theta_{1}}{\operatorname{argmin}} J\left(\theta_{0}, \theta_{1}\right)
$$

where $J(\theta_0,\theta_1)$ is called **cost function** or **square error cost function**

#### b. intuition

1. simplified problem:

![image-20220124185916202](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220124185916202.png)

$\theta_0=0$

dataset: (1,1), (2,2), (3,3)

the plot of function $J$ is on the right. To solve this question, we have to find $\theta_{1}$ s.t. $\dfrac{\operatorname dJ}{\operatorname d \theta_1}=0$

and evidently $\theta_1 =1$

2. original problem:

![image-20220124193051621](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220124193051621.png)

![image-20220124192636409](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220124192636409.png)

3-D figure and it is projected to $\theta_0O\theta_1$ as a **contour plot**. At the current point shown in the figure above, we get the minimum of $J(\theta_0,\theta_1)$, thus find the best $h_{\theta}(x)$ as the hypothesis function.

### 4. Parameter Learning (gradient descent)

#### a. introduction

![image-20220125190359806](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125190359806.png)

go under the hill on the **steepest way**(partial derivative)

property:

![image-20220125191106268](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125191106268.png)

Local optima? Global optima?

![image-20220125190232708](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125190232708.png)

the mathematical implementation of GD

where:

$\alpha$ learning rate: how big a step will be

**WARNING: FOCUS ON THE CORRECT UPDATE VERSION**

what the derivative does:

![image-20220125200155381](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125200155381.png)

what $\alpha$ does:

![image-20220125192057096](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125192057096.png)

if $\theta_1$ is already the optimal minimum, the algorithm just leave it unchanged, so we get:

![image-20220125201124362](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125201124362.png)

#### b. GD for linear regression

the derivative:

![image-20220125202301610](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125202301610.png)

plug back to the GD algorithm:

![image-20220125202501109](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125202501109.png)

An example to calculate this($\theta_1$)

![image-20220125212745890](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125212745890.png)

the cost function of linear regression is always a **convex function** like this

![image-20220125211757164](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220125211757164.png)

### 5. Linear Algebra Fundamentals(Review)

#### a. matrices and vectors(trivial)

#### b. addition and scalar multiplication(trivial)

#### c. matrix matrix multiplication

neat trick to do prediction by one hypothesis function

![image-20220126161808846](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220126161808846.png)

another neat trick in the situation of many hypothesis functions

![image-20220126162733233](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220126162733233.png)

properties:

1. Not commutative;
2. Associative;
3. Identity matrix is commutative;

#### d. inverse and transpose(trivial)

## W2

### 1. Vectorization(intro)

**trick 1**

![image-20220129163953867](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220129163953867.png)

**trick 2**

![image-20220129165218049](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220129165218049.png)

note: treat $\theta$ as a vector, $\Sigma$ blah blah $x$ also a vector.

### 2. Multivariate Linear Regression

***Glossary #2***

$n$: number of features

$x^{(i)}$: the $i^{th}$ entry of the training set

$x^{(i)}_{j}$: the value of feature $j$ of $x^{(i)}$

#### a. hypothesis function of MLR

$h_{\theta}(x)=\theta_0+\theta_1x_1+\theta_2x_2+\theta_3x_3+...+\theta_nx_n$

let $x_0=1$

let $x=\left[\begin{array}{c}
x_{0} \\
x_{1} \\
x_{2} \\
\vdots \\
x_{n}
\end{array}\right]$ 

let $\theta=\left[\begin{array}{c}
\theta_{0} \\
\theta_{1} \\
\theta_{2} \\
\vdots \\
\theta_{n}
\end{array}\right]$

thus we have

$h_{\theta}(x)=\theta^Tx$

#### b. GD for multiple variables

just the same thing...

![image-20220130133928685](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130133928685.png)

#### c. practical tricks for GD of MLR

**1. feature scaling(特征缩放)**

**idea:** make sure features are on the similar scale

![image-20220130195126757](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130195126757.png)

![image-20220130195355992](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130195355992.png)

![image-20220130195940398](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130195940398.png)

**2. learning rate**

![image-20220130201816473](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130201816473.png)

![image-20220130203539767](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130203539767.png)

$\alpha$ can neither be too small nor too big

#### d. Features and Polynomial Regression

![image-20220130211756823](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130211756823.png)

note: treat the $size,size^2,size^3$ as $x_1,x_2,x_3$ and the problem is equivalent to MLR

![image-20220130212116194](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220130212116194.png)

note: an algorithm will help us to choose a feature

**idea:**

1. we can combine various features to one feature
2. we can create new feature
3. we have multiple choices of features

#### e. normal equation(正规方程)

solve $\theta$ analytically

![image-20220131132440915](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220131132440915.png)

**idea:** derivation! derivation!

an example:

![image-20220131133419467](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220131133419467.png)

*choice of GD & NE*

pros and cons of the two methods

![image-20220131135752383](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220131135752383.png)

if $n$ is large(probably $n>10000$), then use GD else use NE

use `pinv()` instead of `inv()` in MATLAB in case of $X^TX$ is non-invertible

## W3

### 1. Classification Introduction

binary-: just$\{0,1\}$

multi-: $\{0,1,2 ...\}$

using Linear Regression is not a good idea in classification problems

### 2. Logistic Regression

#### a. sigmoid function(logistic function)

in classification problems we have:

$h_\theta(x)=g(\theta^Tx)$

where:

$$
g(z) = \frac {1}{1+e^{-z}}
$$

thus, we have

$$
h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}
$$

the diagram of the sigmoid function:

![image-20220204122730048](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220204122730048.png)

the output of $h_\theta(x)$ means **probability**, i.e. $P(y=1|x;\theta)$

![image-20220204123109886](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220204123109886.png)

and we have:

![image-20220204124633204](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220204124633204.png)

#### b. decision boundary(决策边界)

example of logistic regression prediction:

![image-20220204130409252](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220204130409252.png)

decision boundary example: draw a line(decision boundary) to take different classes apart 

![image-20220204141714498](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220204141714498.png)

non-linear decision boundary:

![image-20220204142401348](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220204142401348.png)

#### c. logistic regression cost function & gradient descent

for linear regression we have:

$$
J(\theta)=\frac{1}{2m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}
$$

rewrite it as:

$$
J(\theta)=\frac{1}{m} \sum_{i=1}^{m}\text {Cost}(h_\theta(x),y)
$$

where:

$$
\text {Cost}(h_\theta(x),y)=\frac{1}{2}(h_\theta(x^{(i)})-y^{(i)})^2
$$

but in logistic regression,we have:

$$
\text{Cost}\left(h_{\theta}(x), y\right)=\left\{\begin{aligned}
-\log \left(h_{\theta}(x)\right) & \text { if } y=1 \\
-\log \left(1-h_{\theta}(x)\right) & \text { if } y=0
\end{aligned}\right.
$$

$\log$ is actually $\ln$, the diagram of the cost function(y==1) is:

![image-20220205225106176](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220205225106176.png)

the other part of the diagram is:

![image-20220205225515139](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220205225515139.png)

simplified cost function:

$$
\text{Cost}\left(h_{\theta}(x), y\right)=-y \log \left(h_{\theta}(x)\right)-(1-y) \log \left(1-h_{\theta}(x)\right)
$$

as $y$ just has two possible values, it is equivalent to the previous equation.

so we have the final version of the cost function of logistic regression:

$$
J(\theta)=-\frac{1}{m} (\sum_{i=1}^{m}y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log (1-h_{\theta}(x^{(i)}))
$$

a vectorized version is:

![image-20220206122130470](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220206122130470.png)

*this function is derived from MLE in statistics*

and the GD for logistic regression:

![image-20220206122013814](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220206122013814.png)

just like that in linear regression

#### d. other optimization algorithms

![image-20220206123230145](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220206123230145.png)

how to use it in MATLAB:

step1: write a function that calculates $J(\theta)$ and $\dfrac {\partial J}{\partial \theta_j}$

```matlab
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

step2:

use function `fminunc()` to optimize $\theta$

```matlab
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

### 3. Multi-class Classification

#### a. one-vs-all classification

**idea:** separate this problem into different binary-classification problems

example:

![image-20220207105743692](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220207105743692.png)

i.e. we will train logistic regression classifiers as below:

$$
h_\theta^{(i)}(x) = P(y=i|x;\theta)
$$

where $i=1,2,\dots$

when making predictions, we just have to calculate $\max_i h_\theta^{(i)}(x)$

### 4. Overfitting

#### a. introduction

example:

![image-20220208102656807](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208102656807.png)

2 ways to solve the problem:

![image-20220208104447251](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208104447251.png)

#### b. regularization

1. intuition&idea:

![image-20220208105059948](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208105059948.png)

![image-20220208105555238](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208105555238.png)

2. cost function

$$
J(\theta)=\frac{1}{2 m}\left[\sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right)^{2}+\lambda \sum_{j=1}^{n} \theta_{j}^{2}\right]
$$

where

$$
\lambda \sum_{j=1}^{n} \theta_{j}^{2}
$$

is the regularization term.

$\lambda$ cannot be too large, otherwise it may cause underfitting

3. regularized linear regression

gradient descent:

![image-20220208113156852](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208113156852.png)

normal equation:

![image-20220208121253109](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208121253109.png)

note that $X^TX+\lambda\left[\begin{array}{lllll}
0 & & & & \\
& 1 & & & \\
& & 1 & & \\
& & & \ddots & \\
& & & & 1
\end{array}\right]$ is always invertable

4. regularized logistic regression

gradient descent:

just like that in linear regression, the only difference is the definition of $h_\theta(x)$

![image-20220208133845596](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208133845596.png)

advanced algorithms in MATLAB:

![image-20220208134050970](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220208134050970.png)

just alike.

## W4

### 1. Neural Network: Intro & Representation

#### a. motivation

too many features if we want to train a real classifier like a car detector

brains can actually learn to handle different kinds of signals on the same part once they are connected with different sensors.

![image-20220210121214139](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220210121214139.png)

#### b. model representation

***Glossary #3***

bias unit: input $x_0$

sigmoid activation function: $g(x)=\frac{1}{1+e^{-x}}$

weights: vector $\theta$

![image-20220211110541384](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220211110541384.png)

input layer, hidden layer; output layer: as shown in the figure below:

![image-20220211111726945](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220211111726945.png)

the computational steps:

![image-20220211112445976](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220211112445976.png)
vectorized steps: **forward propagation** 

![image-20220211122451103](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220211122451103.png)

NN just do things like logistic regression except that the features fed to $h_{\theta}(x)$ are from the hidden layers. **NN learn its own features**

#### c. intuition & application

**intuition: XNOR network**

![image-20220212105448938](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220212105448938.png)

starting from AND

![image-20220212105527382](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220212105527382.png)

and OR

![image-20220212105803507](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220212105803507.png)

and NOT

![image-20220212110003098](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220212110003098.png)

finally we can get:

![image-20220212110907374](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220212110907374.png)

**application: handwritten digits classification**

## W5

### 1. Neural Network: Learning

#### a. cost function

***Glossary #4***

$L$: the number of layers of a neural network

$s_l$: the number of units of each layer(without bias unit)

$K$: the dimension of the output layer

**cost function of NN in classification problems:**

![image-20220213131217466](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220213131217466.png)

#### b. back propagation: compute the derivatives of $J(\Theta)$

1. **compute $\delta^{(j)}$**

![image-20220213182013515](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220213182013515.png)

2. **compute derivatives**

![image-20220213182456653](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220213182456653.png)

**steps of BP**

Given training set $\lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace$

1. Set $\Delta^{(l)}_{i,j} := 0$ for all (l,i,j), (hence you end up having a matrix full of zeros)

`for` training example t =1 to m:

1. Set $a^{(1)} := x^{(t)}$
2. Perform forward propagation to compute $a^{(l)}$ for l=2,3,…,L:

![image-20220213192353263](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220213192353263.png)

3. Using $y^{(t)}$, compute $\delta^{(L)} = a^{(L)} - y^{(t)}$

4. Compute $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)}$ using $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .* \ (1 - a^{(l)})$

where $a^{(l)}\ .* \ (1 - a^{(l)})=g'(z^{(l)})$ namely the derivative of the sigmoid function

5. $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$

end of `for` loop

1. $D^{(l)}_{i,j} := \dfrac{1}{m}(\Delta^{(l)}_{i,j}+\lambda\Theta^{(l)}_{i,j})\  \text{if}\ j≠0$.

2. $D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j}\  \text{if}\ j=0$

then we get the partial derivative $\dfrac{\partial}{\partial\Theta^{(l)}_{i,j}}J(\Theta)=D^{(l)}_{i,j}$

#### c. BP in practice

1. **parameter unrolling from matrices to vectors(系数展开)**

reason: advanced optimizing algorithms treat the inputs as vectors but in NN the inputs are matrices

**idea:**  unroll matrices to vectors

example:

![image-20220214120707744](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220214120707744.png)

usage in advanced learning algorithm

![image-20220214131021157](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220214131021157.png)

2. **gradient checking**

reason: to debug

**idea:** check that whether the gradient given by $D^{(l)}$ is similar to the gradient given by its definition(using the formula $\dfrac{J(\theta+\epsilon)-J(\theta-\epsilon)}{2\epsilon}$) 

implementation:

![image-20220214141501196](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220214141501196.png)

notice that we have to re-initialize `thetaPlus` and `thetaMinus` because we just calculate the partial derivative of each $\theta^{(i)}$

advice on how to use gradient check:

![image-20220214141812637](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220214141812637.png)

3. **random initialization**

reason:

![image-20220214155815773](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220214155815773.png)

#### d. summary: how to train a neural network?

**1. pick a network architecture**

![image-20220215145220436](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220215145220436.png)

**2. training through 6 steps**

![image-20220215160939864](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220215160939864.png)

![image-20220215161047994](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220215161047994.png)

## W6

### 1. Evaluating a Learning Algorithm

#### a. test set error

![image-20220218165452443](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220218165452443.png)

training : test $\approx$ 7 : 3

learn through the training set; compute $J_{test}$ of the test set by the formulas below(varies between linear regression and classification):

![image-20220218170813905](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220218170813905.png) 

#### b. model selection & Train/Validation/Test sets

***Glossary #6***

$d$: the degree of the current model

$\theta^{(d)}$: the parameters learnt by the model of degree $d$

The problem is:

**![image-20220218173221565](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220218173221565.png)**

and to solve it, we split the dataset into 3 sets: Training/(Cross) Validation/Test sets instead of 2

training : validation : test $\approx$  6 : 2 : 2

and compute the error of each set

**steps of selecting the model using TVT sets**

1. learn parameters of each model using the training set
2. compute $J_{cv}$ using the validation set and choose the degree that minimizes $J_{cv}$ and select the model
3. estimate generalized performance using the test set

### 2. Bias(偏差) & Variance(方差)

#### a. detection

high bias: underfit: $J_{train}(\theta)$ 

high variance: overfit: $J_{cv}(\theta)$

how to detect:

![image-20220219231610008](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220219231610008.png)

#### b. regularization and bias/variance

how to choose the regularization parameter $\lambda$:

![image-20220220150600491](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220220150600491.png)

note that all the definition of  $J_{train}$ $J_{cv}$ and $J_{test}$ are without the regularization term

#### c. learning curves

**high bias**

![image-20220220172822928](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220220172822928.png)

**high variance**

![image-20220220172853375](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220220172853375.png)

#### d. review: what to do next?

![image-20220220174707346](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220220174707346.png)

### 3. ML System Design

#### a. example: build a spam classifier

**prioritizing what to do next**

![image-20220222161441971](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220222161441971.png)

![image-20220222162928867](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220222162928867.png)

**error analysis**

recommended approach when **building a machine learning system**

1. implement a simple algorithm
2. cross validation
3. list possible approaches to improve it

recommended approach when **doing error analysis**

1. find the specific type of the mistaken test cases

e.g.(spam classifier)

![image-20220222200244248](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220222200244248.png)

2. list possible cues to improve the performance

**doing numerical evaluation** is important

![image-20220222202533390](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220222202533390.png)

#### b. handling skewed data(偏斜数据)

**definition**

the number of data in one class is much greater than that in another class(e.g. only 0.05% of humanities have cancer) and **causes overfit**

**precision(查准率) and recall(召回率)**

![image-20220225131219852](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220225131219852.png)

**tradeoff between precision and recall**

![image-20220225171414316](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220225171414316.png)

**F score**

![image-20220225171735967](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220225171735967.png)

note: choose a proper threshold using **cross validation** set to compute the F score

#### c. collecting data

![image-20220226095858122](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220226095858122.png)

**rationale**

1. sufficient to predict/classify

![image-20220226133110843](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220226133110843.png)

2. parameters of the learning algorithm 

![image-20220226154458698](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220226154458698.png)

a large scale of data can help in both situations

## W7

### 1. SVM(Support Vector Machine)

#### a. large margin classification(大间距分类)

**an alternative viewpoint of logistic regression**

a new definition for the cost function of logistic regression $\text{cost}_1(z)$ (y=1) and $\text{cost}_0(z)$(y=0) as the blue line shows  :

![image-20220302192839854](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220302192839854.png)

definition of SVM:

![image-20220302200206281](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220302200206281.png)

$m$ is removed from the cost function and $C=\dfrac{1}{\lambda}$, and $h_{\theta}(x)$ is replaced by $\text{cost}_{0,1}(\theta^Tx^{(i)})$

**intuition**

We want $\theta^Tx\ge1$ instead of 0 if $y=1$ otherwise we want $\theta^Tx\le-1$ instead of 0 because we want to minimize the cost function by setting the first sum 0:

![image-20220302210123634](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220302210123634.png)

SVM choose the best decision boundary with the largest **margin**(the distance between the boundary and the nearest (to the boundary) points of each class):

![image-20220302213248693](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220302213248693.png)

blue lines are margins and SVM will choose the black one

but in datasets with outliers like this:

![image-20220302213419667](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220302213419667.png)

if $C$ is very large the SVM will choose the magenta line as the boundary; otherwise it will remain choosing the black one.

**math behind LMC**

the SVM tries to solve:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-24-22-20-04-image.png)

i.e. (simplified version)

![image-20220304111427652](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304111427652.png)

the reason why SVM doesn't choose the boundary on the left(seems less 'natural' than that on the left) is: $|p^{(i)}|$ is small so that $\|\theta\|$ shall be **large** but that is **contradictory to the first minimizing condition**

here is why $\theta$ is vertical to the boundary:

![image-20220304115209418](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304115209418.png)

#### b. kernels(adapting SVM to non-linear classification problems)

**Gaussian kernels**

![image-20220306104417445](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220306104417445.png)

​    **example**

![image-20220306110811836](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220306110811836.png)

**how to predict**

![image-20220306112058449](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220306112058449.png)

**choose the landmarks**

just choose $l^{(i)}=x^{(i)}$

![image-20220306113747214](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220306113747214.png)

**combining SVM with kernels**

just modify $\theta^Tx^{(i)}$ to $\theta^Tf^{(i)}$ like this:

![image-20220306115921719](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220306115921719.png)

**parameters**

![image-20220306120158668](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220306120158668.png)

#### c. SVM in practice

in programming we don't need to write the SVM functions ourselves but we have to specify two things below:

- the choice of parameter $C$
- the choice of the kernel(similarity function)
- (when using Gaussian kernel) implement the kernel function

**choice of the kernel**

1. no kernel(i.e. linear kernel): gives a standard linear classifier i.e. predict $y=1$ if $\theta^Tx\ge0$. one situation to use it is there are many features but few training data
2. Gaussian kernel: few features and many training data, **do perform feature scaling**!
3. other choices:

![image-20220307105935889](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220307105935889.png)

**multi-classification using SVM**

![image-20220307111146003](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220307111146003.png)

**logistic regression vs. SVM**

$n$ is the number of features; $m$ is the number of training examples

- $n$ is large relative to $m$, use logistic regression or SVM without kernels
- $n$ small while $m$ intermediate: SVM with Gaussian kernel
- $n$ small while $m$ large: add more features, use logistic regression or SVM without kernels

![image-20220307113615118](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220307113615118.png)

## W8

### 1. Unsupervised learning

what it looks like:

![image-20220311102348038](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311102348038.png)

where it is applied:

![image-20220311103110743](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311103110743.png)

#### a. K-Means algorithm

**steps**

1. randomly choose cluster centroids(聚类中心)

![image-20220311104600942](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311104600942.png)

2. assign the training data into clusters by the distance between the data and the cluster centroids

![image-20220311104915776](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311104915776.png)

3. move the cluster centroid to the 'average' of each cluster

![image-20220311111150395](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311111150395.png)

4. repeat until the centroid remains unchange

more generally, the algorithm goes like this:

![image-20220311143706964](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311143706964.png)

![image-20220311143724153](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311143724153.png)

**Note that if no points were assigned to a cluster, a common way to handle this is eliminating the cluster and modify $K$ to $K-1$. If you really need $K$ clusters, just reinitialize the cluster centroid.**

**optimization objective**

***Glossary #7***

$c^{(i)}$: index of current cluster to which example $x^{(i)}$ is assigned

$K$: number of clusters

$\mu_k$: cluster centroid $k$

$\mu_{c^{(i)}}$: centroid of cluster to which example $x^{(i)}$ is assigned

the optimization objective(**distortion cost function(失真代价函数)**):

![image-20220311150130318](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311150130318.png)

the cluster assignment step is actually optimizing $c^{(i)}$

the cluster centroid step is actually optimizing $\mu_k$

function $J$ won't sometimes increase!

**random initialization**

principles:

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311151245658.png" alt="image-20220311151245658" style="zoom:50%;" />

to avoid local optima, we can **run K-means many times and select the one with the lowest error as the result.**

**choosing $K$**

*Elbow method:*

for k=1 to m(exclusive) run K-Means and compute $J$, correct $K$ shall be the value at the "elbow":

![image-20220311152741579](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311152741579.png)

but if the plot is like this:

![image-20220311152913826](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311152913826.png)

It is hard to find the "elbow"

### 2. Dimensionality Reduction

#### a. motivation

**data compression**

make the algorithm run faster

(2->1)

![image-20220312125909101](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220312125909101.png)

(3->2)

![image-20220312131408306](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220312131408306.png)

**visualization**

helps visualize the data easily

#### b. PCA(Principle Component Analysis)

PCA tries to find a line(super-plain, i.e., $k$ vectors) onto which data is projected where the sum of the **distance** is minimized 

![image-20220313200943945](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220313200943945.png)

![image-20220313202553617](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220313202553617.png)

**PCA is NOT linear regression**

- PCA aims to minimize the distance but LR aims to minimize the y-direction square error
- PCA is not for prediction but data dimension reduce

**steps of the algorithm**

1. data pre-processing: apply mean normalization(must); feature scaling(optional) to all features

![image-20220313214555460](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220313214555460.png)

2. compute covariance matrix and the eigenvectors of the covariance matrix using SVD(Singular-Value Decomposition)

![image-20220313212155464](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220313212155464.png)

The first $k$ columns of the $U$ matrix are exactly $u^{(1)},u^{(2)},\dots u^{(k)}$ and $z=\begin{bmatrix}u^{(1)}&u^{(2)}&\dots&u^{(k)}\end{bmatrix}^Tx$

#### c. applying PCA

**data reconstruction**

as $z=\begin{bmatrix}u^{(1)}&u^{(2)}&\dots&u^{(k)}\end{bmatrix}^Tx$

$x=(\begin{bmatrix}u^{(1)}&u^{(2)}&\dots&u^{(k)}\end{bmatrix}^T)^{-1}z=\begin{bmatrix}u^{(1)}&u^{(2)}&\dots&u^{(k)}\end{bmatrix}x$ (pseudo inverse here)

Here is why $\begin{bmatrix}u^{(1)}&u^{(2)}&\dots&u^{(k)}\end{bmatrix}^T=\begin{bmatrix}u^{(1)}&u^{(2)}&\dots&u^{(k)}\end{bmatrix}^{-1}$

![image-20220315200340291](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220315200340291.png)

**choosing $k$**

principle:

![image-20220315201114191](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220315201114191.png)

algorithm:

![image-20220315202307119](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220315202307119.png)

**application**

1. supervised learning speedup

![image-20220315225912030](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220315225912030.png)

2. **DO NOT** use PCA to prevent overfitting, apply regularization instead
3. **DO NOT** use PCA if unnecessary

## W9

### 1. Anomaly Detection

#### a. density estimation

**motivation**

find out the point that is anomalous:

![image-20220318111209623](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318111209623.png)

**Gaussian distribution(trivial)**

**algorithm**

1. choose probable anomalous features $x_i$
2. fit $\mu_j,\sigma_j^2$ using:

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318112539909.png" alt="image-20220318112539909" style="zoom:50%;" />

3. compute $p(x)$ for a new example $x$ using:

![image-20220318113110562](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318113110562.png)

4. anomaly if $p(x)<\varepsilon$

#### b. building an anomaly detection system

**evaluation metrics**

![image-20220320193128716](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220320193128716.png)

precision/recall/F score(see W6)

Note that **classification accuracy is not used as a metric in case of skewed data**

**anomaly detection vs. supervised learning**

different features:

![image-20220320195906143](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220320195906143.png)

different usages:

![image-20220320201531980](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220320201531980.png)

**choosing what features to use**

handling non-Gaussian features:

![image-20220320203947311](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220320203947311.png)

do some transformations to make it more Gaussian:

1. $x:=\log(x+c)$
2. $x:=x^c$

the constant $c$ is what we adjust.

use `hist(x)` to plot the histogram of the data

**error analysis:**

what we want:

![image-20220320210120148](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220320210120148.png)

the problem is:

![image-20220320210222940](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220320210222940.png)

solution: combine features that will distinguish between normal cases and anomalous cases

#### c. multivariant Gaussian distribution

an example when univariant Gaussian fails to detect the anomaly:

![image-20220321101914083](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321101914083.png)

the green point is anomalous but the Gaussian values of both features are relatively high.

**MGD**

Instead of model $p(x_i)$ separately, we model $p(x)$ in one go.

parameters: $\mu\in\mathbb{R}^n$ $\Sigma\in\mathbb{R}^{n\times n}$, where $\Sigma$ is the covariance matrix

the PDF for MGD:

$p(x;\mu,\Sigma)=\dfrac{1}{(2\pi)^{\frac{1}{2}}|\Sigma|^{\frac{1}{2}}}\exp{(\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))}$

where $x$ is a vector

**examples**

1. varying the diagonal elements of $\Sigma$(**the range of each feature**):

synchronized:

![image-20220321113952500](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321113952500.png)

asynchronized:

![image-20220321114204624](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321114204624.png)

![image-20220321114233720](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321114233720.png)

2. varying off-diagonal elements  of $\Sigma$(**the correlation between each feature**):

![image-20220321114934090](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321114934090.png)

![image-20220321115210252](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321115210252.png)

3. varying $\mu$(**the location of the peak of the function**)

![image-20220321115500619](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321115500619.png)

**anomaly detection using MGD**

steps:

1. fitting parameters using the same approach as PCA.

2. compute the PDF of MGD of the new example:

![image-20220321121143074](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321121143074.png)

relationship to the original model:

![image-20220321121516741](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321121516741.png)

constraints: the off-diagonal elements shall all be 0s

The MGD automatically take the correlations between features

here is more concrete comparisons:

![image-20220321123455728](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321123455728.png)

if $\Sigma$ is non-invertible, then there are 2 possible reasons:

- $m\le n$
- redundant features

### 2. Recommender Systems

#### a. predicting movie ratings(content-based)

***Glossary #8***

$n_u$ : number of users

$n_m$ : number of movies

$r(i,j)$: 1 if user $j$ has rated movie $i$

$y^{(i,j)}$ the rate given by user $j$ to movie $i$ (defined only when $r(i,j)=1$)

recommender system tries to fill in the blanks marked '?' and find out what a user likes

![image-20220322124053647](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220322124053647.png)

**content based recommendations**

idea:

![image-20220322130741444](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220322130741444.png)

**problem formulation**

![image-20220322142317989](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220322142317989.png)

to simplify, $m^{(j)}$ is ignored:

the optimization objective:

![image-20220322143427405](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220322143427405.png)

algorithm: gradient descent again:

![image-20220322143910735](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220322143910735.png)

#### b. collaborative filtering(协同过滤)

**idea**

assumption #1: we have no idea about the features of the movie(this is more realistic)

![image-20220323124220342](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323124220342.png)

assumption #2: users have told us what kind of movie they like. i.e., the $\theta$ parameters are given

![image-20220323124452726](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323124452726.png)

we can infer the value of features from the $\theta$ values and the rates:

![image-20220323143604239](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323143604239.png)

this is essentially the same algorithm as the content-based one.

![image-20220323144228921](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323144228921.png)

the chain is interesting isn't it? But it is not as efficient as it can be. we can solve $\theta$ and $x$ simultaneously by combining the cost functions together.

**optimization objective**

![image-20220323161950670](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323161950670.png)

we ignore the convention that $x_0=1$ because the algorithm can do this itself.

**algorithm**

steps:

1. initialize $x$ and $\theta$ to small random values(**to ensure that $x^{(1)}$ to $x^{(n_m)}$ can be learnt separately**)
2. use GD to minimize $J$

![image-20220323162937709](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323162937709.png)

3. predict!

![image-20220323163119606](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323163119606.png)

#### c. low rank matrix factorization

**vectorization: low rank matrix factorization**

![image-20220324124458945](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220324124458945.png)

**finding related movies**

![image-20220324130707490](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220324130707490.png)

**mean normalization**

if a user hasn't rated any movies, the $\theta$ parameters will eventually go to all zeros because of the regularization term and it is not possible to recommend any movies

![image-20220324132157493](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220324132157493.png)

what we do:

add the mean rate of each movie to the prediction:

![image-20220324133519674](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220324133519674.png)

## W10

### 1. Stochastic Gradient Descent(large scale GD)

#### a. batch GD

just the same as the algorithm in Week 2

![image-20220328110721538](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328110721538.png)

#### b. SGD

**algorithm**

![image-20220328112121521](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328112121521.png)

Observe one training example every time and the model eventually wanders around the global minimum.

**convergence problem**

*convergence checking:*

![image-20220328114847879](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328114847879.png)

4 possible situations:

![image-20220328115146624](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328115146624.png)

1 and 2: no bugs

3: may have to change something in the algorithm or set the average number larger(say, 5000)

4: use smaller learning rate $\alpha$

*an approach to make SGD converge to the global minimum*

slowly decrease $\alpha$ in each epoch. $\alpha$ can be set to $\alpha=\dfrac{\text{const1}}{\text{itertion number}+\text{const2}}$

the meandering will be smaller and smaller while approaching the global minimum:

![image-20220328121118294](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328121118294.png)

#### c. mini-batch GD

**idea**

![image-20220328113627508](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328113627508.png)

**algorithm**

![image-20220328113848183](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328113848183.png)

### 2. Advanced Topics

#### a. online learning

learn from a data stream. In each epoch we just consider one training example:

![image-20220330102538992](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220330102538992.png)

another example: information retrieval

![image-20220330103557336](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220330103557336.png)

#### b. map reduce & data parallelism

**map reduce idea:**

compute the gradient on different computers(or different cores) and combine them in a master server

![image-20220330105232297](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220330105232297.png)

![image-20220330105924284](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220330105924284.png)

## W11

### Case Study: Photo OCR

#### a. pipeline

1. text detection
2. character segmentation
3. character classification
4. (optional) spelling correction

![image-20220402114145446](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220402114145446.png)

#### b. sliding window technique

**pedestrian detection problem**

1. determine the size of the window
2. slide the window from the left top to the right bottom to detect whether there is a pedestrian
3. choose a larger window size(resize it to the image size of the training set, say 80*40) and repeat 1,2.

**text detection problem**

1. use slide window to detect all the text regions in an image:

![image-20220402125816949](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220402125816949.png)

2. the result may be like this

![image-20220402130123715](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220402130123715.png)

3. apply expansion operator:

![image-20220402130205612](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220402130205612.png)

**character segmentation**

train a classifier to determine whether there is **a split between characters**

![image-20220402132157516](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220402132157516.png)

applying SW on this:

![image-20220402132615813](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220402132615813.png)

#### c. getting data for the system

1. artificial data synthesis

![image-20220403120318180](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220403120318180.png)

2. introducing distortions:

![image-20220403125955852](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220403125955852.png)

**advice on getting more data**

![image-20220403220833345](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220403220833345.png)

#### d. ceiling analysis(上限分析)

**examples:**

![image-20220404185921286](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220404185921286.png)

![image-20220404192647165](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220404192647165.png)

Manually make each step of the pipeline to the ground truth and check how much the accuracy improved. 

## End of Course: Thank you from Andrew Ng

![image-20220404194314334](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220404194314334.png)
