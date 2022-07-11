# Dive into DL note

[TOC]

## Part 0: DL Basis

### 1. Introduction

**An AI map:**

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220421155713952.png" alt="image-20220421155713952" style="zoom:50%;" />

**Some fields of DL**

1. image classification:

[ImageNet](http://www.image-net.org):

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422105705735.png" alt="image-20220422105705735" style="zoom:50%;" />

2. object detection and segmentation

[RCNN](https://github.com/matterport/Mask_RCNN)

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422110734414.png" alt="image-20220422110734414" style="zoom:50%;" />

3. [style transfer](https://github.com/zhanghang1989/MXNet-Gluon-Style-Transfer/)(样式迁移)
   
   <img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422110756221.png" alt="image-20220422110756221" style="zoom:50%;" />

4. face composition

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422110929029.png" alt="image-20220422110929029" style="zoom:50%;" />

5. [image generation by description](https://openai.com/blog/dall-e/)

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422112851284.png" alt="image-20220422112851284" style="zoom:50%;" />

6. text generation

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422113341404.png" alt="image-20220422113341404" style="zoom:50%;" />

7. self-driving

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422113436434.png" alt="image-20220422113436434" style="zoom:50%;" />

### 2. Data processing using Pytorch

#### a. data operation

1. high-dimension array sample:

![image-20220424113901699](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424113901699.png)

2. visit elements:

![image-20220424115009971](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424115009971.png)

3. implementation of data operation:

```python
import torch  # not 'pytorch'

# SAMPLES
x = torch.arange(12) # generate an array [0,1,2,...,11]
x.shape # return the shape of x
x.numel() # return the element number of x
x.reshape(3,4) # reshape x to a 3*4 matrix
torch.tensor() # just like np.array()

# alike that in MATLAB
torch.zeros()
torch.ones()
torch.eye()

# fundamental operators see trick 2

torch.cat(arrays:tuple,dim:int) # concatenate tensors in a certain dimension
x.sum(dim:int) # get the sum of all elements of x or in a certain dim

A = x.numpy() # convert from tensor to numpy ndarray
B = torch.tensor(A) # convert from numpy ndarray to tensor

a = torch.tensor([3.5])
a.item() # convert 1*1 tensor to python scalar
```

**tricks**

1. No need to manually compute all the dimensions when using `reshape` we can use -1 to leave the calculation for the computer.

![image-20220424131121057](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424131121057.png)

2. fundamental operators:

![image-20220424185058444](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424185058444.png)

3. broadcasting (prone to misuse)

![image-20220424191633157](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424191633157.png) 

4. saving memory

![image-20220424192510189](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424192510189.png)

do the operation **in place!**

#### b. data pre-processing

**an example on a super simple dataset**

1. handling unknown data

2 approaches: interpolation and delete, we consider interpolation here.

fill the N/A with the average of known data

![image-20220424200937727](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424200937727.png)

for the discrete value, we consider NaN as a class:

![image-20220424201405069](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424201405069.png)

2. convert to tensor

![image-20220424201835134](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424201835134.png)

### 3. Linear Algebra

#### a. theories

1. matrix multiplication is twisting the space:

![image-20220424205658226](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424205658226.png)

2. norms of matrix

![image-20220424210108737](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424210108737.png)

3. symmetric & anti-symmetric

![image-20220424210309959](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424210309959.png)

4. positive-definite:

![image-20220424210441266](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424210441266.png)

5. orthogonal matrix & permutation matrix:

![image-20220424210618359](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424210618359.png)

example of permutation matrix:

![image-20220424210816782](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424210816782.png)

6. eigenvalue & eigen-vector

![image-20220424210943410](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220424210943410.png)

#### b. Pytorch implementation

```python
import torch
# x is a vector
len(x) # python int
x.size() # torch.Size equivalent to x.shape
x.T # transpose of x
y = x.clone() # clone x to y with allocating new memory

# A B are matrices
A*B # Hadamard product

torch.dot() # vectors dot
torch.mv() # matrix-vector multiplication
torch.mm() # matrix-matrix mul

# There is a operator @ in Python 3 which is capable to handle all the situations above

torch.norm() # default compute L2-norm of a vector or the F-norm of a matrix
```

**tricks**

1. sum with dimension kept(set `keepdims=True`) then we can use broadcasting:

![image-20220425122059938](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425122059938.png)

2. cumulative sum: keep dimension, computes the cumulative sum of a tensor

### 4. Calculus

#### a. theories

1. scalar derivative

![image-20220425154719805](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425154719805.png)

2. gradient

numerator layout & denominator layout:

numerator layout is used below

![image-20220425155930812](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425155930812.png)

1. overview

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425160340005.png" alt="image-20220425160340005" style="zoom:50%;" />

2. scalar/**vector**

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425160242447.png" alt="image-20220425160242447" style="zoom:50%;" />

examples

![image-20220425160744483](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425160744483.png)

3. **vector**/scalar

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425161446377.png" alt="image-20220425161446377" style="zoom:50%;" />

4. **vector**/**vector**

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425162405273.png" alt="image-20220425162405273" style="zoom:50%;" />

examples:

![image-20220425162623120](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425162623120.png)

expand to matrix input:

![image-20220425162941309](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425162941309.png)

#### b. auto gradient

1. chain law of derivative

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425164107776.png" alt="image-20220425164107776" style="zoom:50%;" />

example:

1. ![image-20220425164423783](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425164423783.png)

2.

![image-20220425164441603](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425164441603.png)

2. auto gradient

the principle: computation graph(计算图)

![image-20220425165056649](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425165056649.png)

explicitly constructed in TF/MXNet

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425165333694.png" alt="image-20220425165333694" style="zoom:50%;" />

implicitly constructed in torch/MXNet

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425165411298.png" alt="image-20220425165411298" style="zoom:50%;" />

modes of autograd:

FP(Forward Propagation) & **BP(Backward Propagation)**

![image-20220425170900157](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425170900157.png)

**BP actually prune the unnecessary branches!**

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425170329946.png" alt="image-20220425170329946" style="zoom:50%;" />

complexity comparison:

![image-20220425170607767](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425170607767.png)

#### c. Pytorch implementation

```python
import torch
# to compute the gradient, we shall do this to store it:
x = torch.arange(4.)
x.requires_grad_(True)
# OR
x = torch.arange(4.0,requires_grad=True)
# then calculate the gradient
y = 2*torch.dot(x,x) # the function to compute the gradient
y,backward() # BP, the result will be stored in x.grad

# if you want to compute another function, you shall do this first or the gradient will cumulate(add)
x.grad.zero_()

# for non-scalar variants you shall convert it to a scalar using .sum() function(?)
y = x*x
y.sum().backward()

# you can move some results out of the computation graph (make the result not a function of x)
u = y,detach()

# you can compute the gradient even for a Python function
```

### 5. Probability

#### a. theories

1. joint probability

![image-20220426225816446](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426225816446.png)

2. conditional probability

![image-20220426225949202](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426225949202.png)

3. Bayes'  Theorem

![image-20220426230211100](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426230211100.png)

4. marginalization

![image-20220426230436015](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426230436015.png)

5. independence

![image-20220426230730248](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426230730248.png)

6. [Here](https://zh-v2.d2l.ai/chapter_preliminaries/probability.html#conditional-prob-d2) is an example to apply all the things above and show the probability is contrast to the gut feeling
7. expectation & square deviation

![image-20220426232450426](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426232450426.png)

8. **likelihood & MLE**

a. likelihood

likelihood is equal to probability numerically, but the difference is:

![image-20220427110621257](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427110621257.png)

Likelihood is **estimating the parameters** according to the observation of variables. it itself is **still a kind of probability**

Why likelihood is equal to probability numerically?

![image-20220427161108451](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427161108451.png)

So we can also consider likelihood above as a function and draw the graph below:

![image-20220427162835567](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427162835567.png)

b. MLE(Maximum Likelihood Estimation)

![image-20220427163805000](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427163805000.png)

Actually as far as I know, linear regression and logistic regression are essentially MLE problems

### 6. Linear Regression

#### a. theories

1. linear model:

![image-20220426212348978](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426212348978.png)

it can be considered as a 1-layer NN:

![image-20220426212514519](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220426212514519.png)

2. normal distribution and MSE

**Why MSE can be the loss function for linear regression**

As linear regression is an MLE problem, we assume there are noises among observation:

$$
y = \mathbf{w}^\top \mathbf{x} + b + \epsilon
$$

where $\epsilon \sim \mathcal{N}(0, \sigma^2)$

We can write the likelihood of $y$ given an $\mathbf x$

$$
P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right)
$$

as $\epsilon=y-\mathbf w^\top \mathbf x-b$

the optimistic values of $\mathbf w$ and $b$ are:

$$
\operatorname*{argmax}_{\mathbf w,b}P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)})
$$

where $\mathbf x^{(i)}$ and $y^{(i)}$ represents an example in the dataset.

convert it to minimized logarithm likelihood:

$$
\operatorname*{argmin}_{\mathbf w,b}-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2
$$

as the first item and $\dfrac{1}{2\sigma^2}$ is independent with $\mathbf w$ and $b$, the objective above is equivalent to MSE in linear regression

*This is also the reason why logistic regression uses cross entropy as the loss.*

3. basic optimization algorithms

a. gradient descent

basic steps:

![image-20220427221617966](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427221617966.png)

mini-batch GD(default optimization algorithm in GD)

![image-20220427223700454](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427223700454.png)

#### b. Pytorch implementation from scratch

from scratch:

1. dataset generate & read
2. model definition
3. loss definition
4. initialize parameters

all implemented by yourself

```python
import torch
torch.normal(mean, std, size:tuple) # return a normal distribution
with torch.no_grad() # disable gradient computation, saving memory (make sure you will not invoke backward())
```

#### c. Pytorch concise implementation

all those above not implemented but using torch APIs

```python
from torch import nn
from torch.utils import data

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

net = nn.Sequential(nn.Linear(2, 1)) # linear network Sequential is a list of NN layers

net[0].weight.data.normal_(0, 0.01) # initialize the weight by normal distribution
net[0].bias.data.fill_(0) # initialize bias

loss = nn.MSELoss() # loss
trainer = torch.optim.SGD(net.parameters(), lr=0.03) # optimizer

# training
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step() # update parameters
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
```

### 7. Softmax Regression

#### a. theories

1. from regression to classification

![image-20220503223643158](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220503223643158.png)

2. encoding

a. for gt:

![image-20220503225148399](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220503225148399.png)

b. for prediction:

![image-20220503224308347](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220503224308347.png)

3. cross entropy

![image-20220503225742465](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220503225742465.png)

**MLE deduction**

1. Still, from the original form of MLE,  we can write the likelihood as:

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
$$

2. To maximize the likelihood, we can minimize the negative logarithm likelihood:

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)})
$$

where:

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j.
$$

and:

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}
$$

3. rewrite optimization objective:

$$
l(\mathbf{y}, \hat{\mathbf{y}}) =  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.

$$

4. gradient of optimization objective

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j
$$

#### b. fundamentals of information theory

1. entropy

quantified information of a certain distribution $P$ is:

$$
H(P) = \sum_j - P(j) \log P(j)
$$

one of the basic theorems in information theory points out that to encode a randomly sampled data from a distribution $P$ , at least $H(P)$ 'nats' = $\dfrac{1}{\ln(2)}$ bits

2. what is cross entropy actually?

![image-20220504110450425](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220504110450425.png)

#### c. commonly used loss functions

1. L2 loss

![image-20220504123605693](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220504123605693.png)

2. L1 loss

![image-20220504125434740](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220504125434740.png)

3. Huber's robust loss

![image-20220504152654297](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220504152654297.png)

#### d. Pytorch from scratch

1. dataset Fashion-MNIST

![image-20220504164643787](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220504164643787.png)

2. softmax implementation

what we actually do: **feed the softmax function with linear model**

a. read the fashion-MNIST dataset

b. flatten the image, regard each pixel as a feature (lose some spatial info, can be solved by CNN)

c. initialize the parameters

```python
import torch
num_inputs = 784 # 28*28
num_outputs = 10 # 10 classes

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

d. **define model and loss:**

```python
# define the model
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # broadcast


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# define loss

# select elements
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y] # this is a new method to select elements

# the cross entropy
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
```

e. define the accuracy

```python
def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
```

e(extra)

for any net we can evaluate the accuracy:

```python
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

the `Accumulator` class is as below:

```python
class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
```

f. **train the model**

```python
def train_epoch_ch3(net, train_iter, loss, updater):  #@save  1 epoch
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

# the optimizer

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save n epochs
    """训练模型（定义见第3章）"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
```

![image-20220505223013697](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220505223013697.png)

g. prediction

```python
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
```

#### e. Pytorch concise

1. initialize the parameters

```python
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights) # run init_weights in all the layers of net
```

2. define the loss & optimizers

```python
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```

3. train the model

```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

![image-20220505223047988](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220505223047988.png)

#### f. re-estimate the implementation of scratch softmax

$\hat y_j = \dfrac{\exp(o_j)}{\sum_k \exp(o_k)}$ may overflow if $o_k$ is large

A trick to solve this is substract $\max(o_k)$ from all $o_k$, this won't change the result of softmax:

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}
\end{aligned}

$$

but if $o_j-\max(o_k)$ is too small, it may underflow, but if we consider cross entropy together with this, the problem will be solved as $\log(\exp(\dots))$ is counteraction.

Thus:

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}
\end{aligned}
$$

### 8. MLP(Multi-Layer Perceptron)

#### a. theories

1. MLP

![image-20220508215529993](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220508215529993.png)

![image-20220508101400223](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220508101400223.png)

how to train an original perceptron

![image-20220508120303605](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220508120303605.png)

convergence theorem for MLP

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220508145914352.png" alt="image-20220508145914352" style="zoom:50%;" />

a linear MLP **cannot solve XOR problems**

![image-20220508213607436](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220508213607436.png)

### b. MLP from scratch

The training method is all the same as that of softmax, the only difference is the definition of model

a. **definition of ReLU**

```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

b. **definition of model**

```py
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1)  # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
```

c.  **definition of loss**

We still use cross-entropy here

```py
loss = nn.CrossEntropyLoss(reduction='none')
```

### c. Pytorch concise

前のパートと同じ、ただモデルの定義は違います：

```python
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
```

もう1つリニア層が増えて、ReLU層も増えています。

### 9. Model Selection

The main theoretical issues can be found in notes of Shuang Liang and Andrew Ng

#### a. Underfit & Overfit

**the capacity(complexity) of the model**

the capacity of the model shall be initially large enough and then decrease this (by weight decay or dropout) to get the lowest generalization error

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-26-22-28-52-image.png)

The circled '1' s are biases.

*VC dimension*

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-26-22-33-02-image.png)

some results

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-26-22-34-37-image.png)

not commonly used in deep learning, as this is not accurate and hard to compute

### 10. Regularization Skills

#### a. Weight Decay

1. hard limit

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-26-22-54-10-image.png)

2. soft limit

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-26-22-57-26-image.png)

3. visualization of affect of penalty to the optimization

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-26-23-07-39-image.png)

4. update of the params

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-26-23-10-50-image.png)

5. Pytorch implementation

(1) from scratch

Firstly, define the L2-penalty:

```python
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2 
```

Then add L2-penalty to the loss:

```python
# 增加了L2范数惩罚项
# 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
l = loss(net(X), y) + lambd * l2_penalty(w)
```

where lambda is a tunable hyper-parameter

L1 performs better than L2:

L2:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-27-15-17-08-image.png)

L1:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-27-15-19-34-image.png)

(2) concise

the penalty is in the trainer:

```py
 # 偏置参数没有衰减
trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
```

typically it is faster than that is implemented from scratch

#### b. Dropout

1. theories

It is essentially adding noises between layers

In standard dropout regularization, the noises is:

$$
\begin{aligned}
h' =
\begin{cases}
    0 & \text{ probability is } p \\
    \dfrac{h}{1-p} & \text{otherwise}
\end{cases}
\end{aligned}
$$

The expectance of $h^\prime$ is equal to that of $h$

Dropout is usually used at the output point of hidden layers:

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-27-15-57-12-image.png" title="" alt="" width="329">

The visualization

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-27-15-55-10-image.png)

In inference, dropout outputs the same thing as the input.

**Dropout is used only in training!**

2. Pytorch implementation

(a) from scratch

Firstly, we implement a `dropout` layer:

```py
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    # to determine which is dropped
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)
```

 Then we define a net structure trained by FashionMNIST

```py
dropout1, dropout2 = 0.2, 0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
```

(b) concise

Just add a `Dropout` layer:

```py
net = nn.Sequential(nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        # 在第一个全连接层之后添加一个dropout层
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        # 在第二个全连接层之后添加一个dropout层
        nn.Dropout(dropout2),
        nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)
```

### 11. Example of A Computation Graph for BP

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-27-16-43-50-image.png)

This graph shows how BP is performed.

### 12. Data Stability & Model Initialization & Activation Function

#### a. gradient boom & vanish

1. boom

This will be very large

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-11-05-54-image.png)

This will be very large

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-11-07-54-image.png)

problems caused by gradient boom

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-12-08-40-image.png)

2. vanish

if we use sigmoid as the activation function:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-12-11-11-image.png)

at the endpoints of the gradient, the value is small

problems:

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-28-12-14-00-image.png" title="" alt="" width="519">

We need to alleviate these problems:

### b.  weight initialization & activation function

some approaches to solve questions above:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-12-18-01-image.png)

The third one is the topic of this part.

Need to: regard the output and the weight of each layer as a random variant, the expectance=0 and the variance is constant

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-12-22-41-image.png)

Approaches:

1. weight initialization

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-28-12-48-04-image.png" title="" alt="" width="480">

example:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-12-57-20-image.png)

The forward variance:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-28-15-29-56-image.png)

backward expectance & variance

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-28-15-42-13-image.png" title="" alt="" width="590">

Xavier initialization:

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-28-17-07-01-image.png" title="" alt="" width="520">

2. activation function

example: linear activation function

forward:

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-29-10-55-47-image.png" title="" alt="" width="518">

backward:

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-29-11-08-04-image.png" title="" alt="" width="493">

using the results above, we can check the activation functions commonly used (by Taylor expansion):

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-06-29-11-13-28-image.png" title="" alt="" width="523">

It can be seen that the results cannot be satisfied with sigmoid function, so we can scale it to $4*\text{sigmoid}(x)-2$

### 13. Kaggle: House Price Prediction

#### a. data pre-processing

continuous data: normalization

for Kaggle competitions, we can concatenate the train and test data and do the normalization:

```py
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

then do  the normalization

```py
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

discrete data: one-hot encoding

```py
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

#### b. training

a classic template for DNN training:

1. define the model and the loss:

```py
loss = nn.MSELoss()
in_features = train_features.shape[1] # number of feats

def get_net():
    net = nn.Sequential(nn.Linear(in_features,256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256,128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128,1))
   return net
# The true loss used in house price prediction
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()
```

2. define the training function

```py
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # the core of training a DL model
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

#### c. cross validation

We use K-fold cross validation in this competition:

first we should separate the training data into train set and cross validation set

```py
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size) # 
        X_part, y_part = X[idx, :], y[idx]
        # valid part
        if j == i:
            X_valid, y_valid = X_part, y_part
        # train part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
```

then perform K-fold:

```py
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k
```

### 14. Competition: California House Price Prediction Recap

Using MLBox to perform AutoML:

```py
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
paths = ['/content/train.csv','/content/test.csv']
target_name = 'Sold Price'

# split the test and train sets
data = Reader(sep=",").train_test_split(paths,target_name)

# drop the drifting feats
data = Drift_thresholder().fit_transform(data)

# the loss
log_rmse = make_scorer(lambda y_true, y_pred: np.sqrt(np.sum((np.log(y_true) - np.log(y_pred))**2)/len(y_true)), greater_is_better=False, needs_proba=False)

# the optimizer
opt = Optimiser(scoring = log_rmse, n_folds = 5)


# the searching space
space = {

        'est__strategy':{"search":"choice","space":["LightGBM"]},  
         'ne__numerical_strategy' : {"space" : [0, 'mean']},
         'fs__strategy':{"search":"choice","space":['l1','variance','rf_feature_importance']},  
        'est__n_estimators':{"search":"choice","space":[150]},    
        'est__colsample_bytree':{"search":"uniform","space":[0.8,0.95]},
        'est__subsample':{"search":"uniform","space":[0.8,0.95]},
        'est__max_depth':{"search":"choice","space":[5,6,7,8,9]},
        'est__learning_rate':{"search":"choice","space":[0.07]} 

        }

params = opt.optimise(space, data,15) # 15 is the searching epochs

# prediction
prd = Predictor()
prd.fit_predict(params, data)

# generate submission file
submit = pd.read_csv("/content/sample_submission.csv",sep=',')
preds = pd.read_csv("save/"+target_name+"_predictions.csv")

submit[target_name] =  preds[target_name+"_predicted"].values

submit.to_csv("submission.csv", index=False)
```

The searching space can be modified and some feature engineering can be done to improve this.

The best score now is 0.21744

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-07-01-23-07-15-image.png" title="" alt="" width="421">

#### 2. other methods

1. AutoGluon (integrated learning)

A detailed introduction can be found in the appendix

2. h2o

The Kaggle notebook: [AutoML(Using h2o) | Kaggle](https://www.kaggle.com/code/wuwawa/automl-using-h2o/notebook)

3. random forest

The Kaggle notebook: [The 4th place approach (Random Forest) | Kaggle](https://www.kaggle.com/code/jackzh/the-4th-place-approach-random-forest/notebook)

#### 3. handling text features

word2vec OR bag-of-words OR transformers, which will be mentioned in next lectures

#### 4. handling dynamic features

TBD

#### 5. about autoML

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-02-20-57-07-image.png)

## Part 1: CNN

### 1. Pytorch Basis of NN

#### a. Model Construction

**1. layers and blocks**

from small to big:

layer->block->model

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-03-12-20-51-image.png)

**2. another implementation of self-defined MLP**

We need to provide 5 basic functions in this part:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-03-12-28-59-image.png)

```py
class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10)  # 输出层 function 4

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X): # function 1
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X))) # function 2
```

**3. `Sequential` block implementation**

```py
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X
```

**4. execute other code in `forward()`**

```py
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和mm函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X) # weight sharing
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
```

**5. mixed model**

```py
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```

### b. Parameter Management

**1. get the parameters**

The parameters can be obtained by `state_dict()`

```py
print(net[2].state_dict())
```

or can be directly visited

```py
print(type(net[2].bias)) # torch.nn.parameter.Parameter
print(net[2].bias) # a parameter instance
print(net[2].bias.data) # the value of the parameter
```

The parameters can also be visited all-in-one by

```py
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

We can also visit the parameters by:

```py
net.state_dict()['2.bias'].data
```

We can get the parameters from a nested model by:

```py
# model definition
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

# get the param
rgnet[0][1][0].bias.data
```

**2. parameter initialization**

Gaussian distribution initialization:

```py
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01) # Gaussian 
        nn.init.zeros_(m.bias) # zeros
net.apply(init_normal)
```

constant initialization(not recommended in practice)

```py
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1) # constant
        nn.init.zeros_(m.bias)
net.apply(init_constant)
```

Xavier initialization

```py
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
net.apply(xavier)
```

customized initialization

```py
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
```

The code above initialized the weight with the distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

Also we can directly modify the parameters:

```py
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```

The weight of layers can be shared by:

```py
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

Shared weights is helpful to reduce the number of parameters

#### c. Customized Layer

**1. without parameters**

```py
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

**2. with parameter**

```py
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```

#### d. File R&W

**1. tensor R&W**

```py
x = torch.arange(4)
torch.save(x, 'x-file')  # W
x2 = torch.load('x-file') # R
```

**2. model parameter R&W**

```py
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)


torch.save(net.state_dict(), 'mlp.params') # W


clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval() # R
```

#### e. about GPUs

**1. check the information of GPU**

CPU: `torch.device('cpu')`

GPU: `torch.device('cuda')`

the $i$th GPU: `torch.device('cuda:i+1')`

check the number of GPU: `torch.cuda.device_count()`

**2. tensor on GPU**

The tensor is stored on CPU by default. It can be checked by `x.device` where `x` is a tensor

We can store it on GPU by explicitly assign this in the code:

```py
X = torch.ones(2,3,device='cuda:0')
```

If there are more than 1 GPUs, say 2. We can store another tensor on GPU 2 by

```py
Y = torch.rand(2,3,device='cuda:1')
```

If you wanna do the operation `X+Y`, you need to copy `X` to `cuda:1` or `Y` to `cuda:0`

```py
X.cuda(1)
# OR
Y.cuda(0)
```

**3. model on GPU**

Similarly, we can store models on GPU by:

```py
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device='cuda:0')
```

Then all the learning process will be done on GPU.

**4. GPU choosing and buying**

There are 2 important things to consider:

1. Graphic memory

2. The computational speed

The performance-price diagram:

![../_images/flopsvsprice.svg](https://zh-v2.d2l.ai/_images/flopsvsprice.svg)

Seems 1080Ti is the most valuable one

The energy consumption-price diagram:

![../_images/wattvsprice.svg](https://zh-v2.d2l.ai/_images/wattvsprice.svg)

### 2. Convolutional Neural Network(CNN)

#### a.  why convolution

The problem of MLP

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-06-10-14-17-image.png)

2 principles:

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-07-06-10-20-29-image.png" title="" alt="" width="500">

from MLP to convolutional layer

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-06-10-25-04-image.png)

expansion: the weight from 2-D to 4-D to ensure every one of the pixels has its corresponding hidden unit.

**using principle #1: translation invariant**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-06-10-31-47-image.png)

This makes the weight shared among some elements

**using principle #2**

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-07-06-11-22-55-image.png" title="" alt="" width="407">

where: $v_{a,b}$ is the convolutional kernel

Thus, by applying principles #1 and #2 to the FC layer, we get the convolutional layer(cross-correlation layer)

#### b. convolutional layer

**convolution calculation**

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-07-06-11-59-31-image.png" title="" alt="" width="329">

```py
def corr2d(X, K):  #@save
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```

**2-D convolutional layer**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-06-12-02-57-image.png)

**examples**

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-07-06-12-13-43-image.png" title="" alt="" width="452">

**cross-correlation vs. convolution**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-06-12-15-37-image.png)

The circled part is flip part of convolution

**1-D and 3-D convolution**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-06-12-17-21-image.png)

*kernel and bias are learnable parameters*

**code: edge detection learning**

```py
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8)) # input
Y = Y.reshape((1, 1, 6, 7)) # gt
lr = 3e-2  # lr

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2 # loss
    conv2d.zero_grad() # clear the gradient
    l.sum().backward() # BP
    conv2d.weight.data[:] -= lr * conv2d.weight.grad # GD
    if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
```

#### c. padding, stride, multi-channels and pooling

**padding: keep the size of output unchanged**

output size:

$$
(n_h-k_h+p_h+1)\times(n_w-k_w+p_w+1)
$$

where:

$n_h,n_w$ the input height and width

$k_h,k_w$ the kernel height and width

$p_h,p_w$ the padding height and width

We need to set $p_h=k_h-1$ and $p_w=k_w-1$ to keep the size of output

Usually $k_h,k_w$ are odd and the padding method is to add $p_h/2$ rows at the top and bottom, $p_w/2$ columns at the left and right

If $k_h$ or $k_w$ is even, one possible situation is to add $\lceil p_h/2\rceil$ and $\lfloor p_h/2\rfloor$ rows at the top and bottom, $\lceil p_w/2\rceil$ and $\lfloor p_w/2\rfloor$ columns at the left and right.

**stride: rapidly reduce the number of parameters**

output size:

$$
\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor
$$

where:

$s_h.s_w$ the stride height and width

so most of the time when the kernel is square the output size can be calculated by:

$$
\dfrac{N-F}{\text{stride}}+1\\

$$

where $N$ is the padded 

**channels**

*input channels*

If the number of channels is bigger than 1, say, $c_i$, then the size of convolutional kernel is $c_i\times k_h\times k_w$. And the output is still: 

$$
\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor
$$

The convolution is computed by:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-09-10-00-43-image.png)

namely:

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-07-09-10-58-38-image.png" title="" alt="" width="610">

It can be implemented by:

```py
def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

*output channels*

The size of kernel: $c_o\times c_i \times k_h \times k_w$

The size of output:

$$
c_o\times \lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor
$$

namely

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-09-11-01-48-image.png)

This can be implemented by:(using `torch.stack()`)

```py
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Different output channels is capable of recognize different patterns:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-09-11-13-44-image.png)

(6 channels, 6 patterns)

and input channels combine them together with weight

_1*1 convolution_

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-09-11-31-24-image.png)

functions of 1*1 convolution:

1. cross-channel feature combination

2. modify the dimension of the feature map

3. more non-linear(especially using ReLU after 1*1 convolution)

This can be implemented by:(by matrix multiplication)

```py
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))
```

*summary*

The size of each param:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-09-11-49-41-image.png)

**ONLY the number of output channel** is a hyper-parameter of convolutional layer

The hyper-parameters above can be tuned with the constructor of `nn.conv2d`

```py
conv = nn.conv2d(in_channels,out_channels,kernel_size,padding,stride)
```

**pooling layer**

*problem of convolution*

convolution is sensitive with position:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-10-09-09-35-image.png)

*max-pooling*

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-10-09-14-15-image.png)

*how pooling solves the problem of translation-invariant*

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-10-09-35-47-image.png)

Pooling is tolerant of 1-pixel translation in this example

*parameters of pooling*

hyper-parameters: window size; stride; padding

learnable parameters: **None**

the size of output = the size of input

<img src="file:///C:/Users/CharlesGao/AppData/Roaming/marktext/images/2022-07-10-09-49-28-image.png" title="" alt="" width="488">

*average-pooling*

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-10-09-53-24-image.png)

This can be implemented by:

(from scratch, without padding & stride)

```py
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y
```

OR (from framework,**the stride is equal to the pooling window size by default**)

```py
pool = nn.MaxPool2d(kernel_size,stride,padding)
```

a typical framework of CNN:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-11-11-18-08-image.png)

### 3. Modern CNN

#### a. LeNet

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-07-11-10-59-23-image.png)

This can be implemented by:

```py
lenet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

For now, we shall evaluate the performance on GPU:

```py
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
```

and the training function shall be slightly modified:

```py
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```







## Appendix

### 1. AutoGluon

https://www.bilibili.com/video/BV1rh411m7Hb/