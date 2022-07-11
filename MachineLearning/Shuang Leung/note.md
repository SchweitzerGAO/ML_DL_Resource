# Shuang Leung ML note

[TOC]

## 1. Introduction & Intuition

How do human make decisions?

1. **trained** from experience
2. **learn** to make good decisions

AI(goal)->pattern recognition(task,70/80s)->machine learning->deep learning

#### a. history

60-70s: reasoning phase

late 70s-80s: knowledge phase

80s-now: learning phase

![image-20220304200455071](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304200455071.png)

**Connectionism-Phase 1**

Neural Network

![image-20220304201659736](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304201659736.png)

**Statistic**

SVM

![image-20220304201804916](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304201804916.png)

**Connectionism-Phase 2**

![image-20220304201857937](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304201857937.png)

****

**trend**

![image-20220304203413397](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304203413397.png)

**challenge**

![image-20220304203513169](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304203513169.png)

#### b. framework & map

![image-20220304205137525](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304205137525.png)

**supervised learning(data,label)**

1. regression: output scalars

2. classification: output class
   
   2.1 binary classification
   
   2.2 multi-class classification

3. structured learning(beyond classification & regression e.g. ASR, machine translation):

![image-20220304211742684](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304211742684.png)

**classification methods**

1. linear

2. non-linear
   
   2.1 deep learning
   
   2.2 SVM, decision tree, KNN...

**non-supervised learning**

1. semi-supervised(some labeled,some not)
2. weak-supervised(incompletely labeled)
3. transfer learning(problems alike)
4. unsupervised learning:

![image-20220304212332303](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304212332303.png)

![image-20220304212423296](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220304212423296.png)

**reinforced learning**

learning from critics(feedbacks)

![image-20220318194006884](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318194006884.png)

**roadmap**

![image-20220318194225167](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318194225167.png)

## 2. Mathematical Fundamentals

![image-20220311192837151](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311192837151.png)

**conditional probability**

![image-20220311192908693](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311192908693.png)

**Total probability formula -> prior probability**

![image-20220311193417953](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311193417953.png)

reason -> result

**Bayes formula-> posterior probability**

![image-20220311193930855](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311193930855.png)

result->reason

**prior probability and posterior probability**

![image-20220311194600475](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311194600475.png)

**independent event**

![image-20220311195607802](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311195607802.png)

**random variables**

**discrete**

![image-20220311200256075](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311200256075.png)

**continuous**

![image-20220311200357342](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311200357342.png)

**numeric features**

**expectation**

**variance**

![image-20220311204926612](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311204926612.png)

**extremum**

**Hessian matrix**

![image-20220311205158276](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311205158276.png)

**Lagrange multiplier**

![image-20220311205651184](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220311205651184.png)

## 3. Model Selection

**Terminologies**

![image-20220318202332073](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318202332073.png)

Note that **sample/instance(样本)** is without labels while **example(样例)** is with labels.

****

**Tasks**

![image-20220318203423763](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318203423763.png)

**Accuracy & Error**

![image-20220318205537307](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318205537307.png)

![image-20220318204936957](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318204936957.png)

**Overfitting & Underfitting**

![image-20220318210157166](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318210157166.png)

solutions:

1. more training data
2. constrained model:

![image-20220318211012601](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318211012601.png)

no excessive constraint!

**General Performance Evaluation**

1. Hold-out

![image-20220318212021170](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318212021170.png)

using **layer sampling(分层采样)** to ensure the consistency of data distribution

**Weakness**

![image-20220318212552644](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220318212552644.png)

2. Cross Validation

![image-20220325191937942](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325191937942.png)

3. n-fold validation

![image-20220325192406280](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325192406280.png)

advantage:

![image-20220325193132884](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325193132884.png)

4. bootstrapping

![image-20220325193949903](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325193949903.png)

**parameter tuning**

![image-20220325194508491](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325194508491.png)

**performance measure**

1. regression:

MSE:

![image-20220325200743717](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325200743717.png)

2. classification:

*Error & Accuracy*

![image-20220325200835291](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325200835291.png)

*Precision & Recall*

![image-20220325201110478](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325201110478.png)

the 2 values are contradictory:

![image-20220325201935596](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325201935596.png)

We want P and R both high.

*F1 Score*

![image-20220325202357637](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325202357637.png)

F1 score is essentially harmonic mean

F-beta score is essentially weighted harmonic mean

*ROC*

![image-20220325202859621](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325202859621.png)

We want TPR high while FPR low

![image-20220325203210732](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325203210732.png)

the variant of the ROC curve is the **threshold**:

![image-20220325204810821](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325204810821.png)

the variant of PR curve is **R**(How to artificially control it?)

*Cost-sensitive Error Rate*

![image-20220325205830335](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325205830335.png)

**bias and variance**

*bias*

![image-20220325211501690](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325211501690.png)

*variance*

![image-20220325212013386](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325212013386.png)

"Visualization" of bias and variance

![image-20220325212240457](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325212240457.png)

Variance & Bias in many models:

![image-20220408200752570](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220408200752570.png)

![image-20220408201151238](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220408201151238.png)

What to do with bias & variance?

![image-20220408201645434](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220408201645434.png)

tradeoff between bias & variance:

![image-20220408202747959](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220408202747959.png)

## 4. Linear Regression & GD

**linear regression**

Step1: select function model

![image-20220408211622566](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220408211622566.png)

Step 2: define the loss function

- MAE
- MSE

MAE is what I haven't seen but more intuitive

![image-20220408212350464](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220408212350464.png)

the contour plot of the loss function

![image-20220408212615586](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220408212615586.png)

**gradient descent**

![image-20220415195330350](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415195330350.png)

for more parameters($w$ and $b$):

![image-20220415195907471](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415195907471.png)

*learning rate:*

![image-20220415200627711](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415200627711.png)

*adaptive LR*

![image-20220415201321638](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415201321638.png)

*SGD*

![image-20220415202646220](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415202646220.png)

*Feature Scaling*

![image-20220415202616901](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415202616901.png)

![image-20220415203125902](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415203125902.png)

*Limitations*

![image-20220415203348432](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415203348432.png)

for 1D function we can compute **the 2-degree derivative** to determine whether it is saddle point

**regularization**

*Ridge Regression*

(2-norm regularization)

![image-20220415210237589](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415210237589.png)

*LASSO Regression*

(1-norm regularization)

![image-20220415210406518](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415210406518.png)

*what L1 and L2 regularization implies:*

![image-20220415210722966](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220415210722966.png)

L2: **evenly distribute** weights among features

L1: **select some of** the features

## 5. Logistic Regression

**type of classifiers**

![image-20220422193314922](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422193314922.png)

*Discriminative:*

conditional probabilities;

doesn't care about how the data is distributed(easier)

*Generative*

union probabilities

cares about how the data is distributed(harder)

At the same scale, **discriminative approach performs better than generative model**, as at most of the time, it is very difficult to solve the distribution of the data

**logistic regression**

1. mapping $[-\infin,+\infin]$ to $[0,1]$

using sigmoid function:

![image-20220422195352468](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422195352468.png)

properties:

![image-20220422195653486](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422195653486.png)

2. loss function

![image-20220422200719603](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422200719603.png)

where $L(w,b)$ is the likelihood of all the data in set $C_1$

We want to maximize $L(w,b)$

after some mathematical transformation:

1. to maximize $L(w,b)$ is equivalent to minimize $- \ln L(w,b)$ as logarithm converts multiplication to add, making the algorithm faster
2. We apply a trick below to make derivation easier as there are just 2 possible prediction values in bi-classification problem:

![image-20220423204115978](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220423204115978.png) 

3. we get the **cross entropy function** to be minimized:

![image-20220422201638059](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422201638059.png)

3. optimization(GD)

gradient of cross entropy*(to be filled)*:

To apply GD for the cross entropy we have to compute the gradient: $\dfrac{\partial L}{\partial w}$

for the first part:

![image-20220423215843559](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220423215843559.png)

for the second part:

![image-20220423220124788](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220423220124788.png)

putting them together:

![image-20220423220153918](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220423220153918.png)

***The same form as Linear Regression!***

4. why not MSE?

![image-20220422204501064](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422204501064.png)

if the estimation is **far away from the target, the gradient of MSE is still 0**, which is we don't want to see.

**Logistic regression vs. linear regression**

![image-20220422205151380](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422205151380.png)

**limitations**

1. cannot solve XOR problem(leave it for NN)

![image-20220422210023451](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422210023451.png)

motivation of NN: cascading sigmoid functions to solve XOR problem

![image-20220422210827235](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220422210827235.png) 

## 6. KNN

**K nearest neighbors definition**

![image-20220506192408928](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506192408928.png)

**steps**

1. find $K$ nearest neighbors

![image-20220506192524488](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506192524488.png)

*definition of the distance*

![image-20220506193230186](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506193230186.png)

2. choose the class

![image-20220506193628587](D:\COURSE_WORK_Bachelor\ST2022Spring\assignments\A2\A2.2\image-20220506193628587.png)

**hyper-parameter fine-tuning**

1. $K$

![image-20220506194539929](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506194539929.png)

2. distance definition

![image-20220506194630610](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506194630610.png)

3. How to choose the hyper-parameters?

K-fold cross validation

![image-20220506195520403](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506195520403.png)

**analysis of KNN**

bias & variance

![image-20220506200550558](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506200550558.png)

![image-20220506201307886](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506201307886.png)

Complexity

![image-20220506201545255](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506201545255.png)

**limitation**

Can we use KNN on images?

![image-20220506202747483](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506202747483.png)

![image-20220506202816730](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506202816730.png)

**When to use KNN?**

1. spatial correlation

*Recommender system*

2. low-dimension data

*Text mining*

## 7. Decision Tree

**basic concept**

![image-20220513191451574](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513191451574.png)

**building a decision tree**

![image-20220506210055736](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506210055736.png)

**the key step**

define a metric to measure the **purity** of an attribute

![image-20220506211110951](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506211110951.png)

![image-20220506211240256](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506211240256.png)

*Entropy: a metric to measure **impurity** & **uncertainty***

1. **entropy is the measurement of uncertainty**
2. **information is the amount of decrement of uncertainty**

example of entropy:

![image-20220506212322560](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506212322560.png)

the varying of entropy

![image-20220506212501513](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506212501513.png)

**information gain** (used by ID3)

![image-20220513192511805](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513192511805.png)

$\text{Ent}(D)$ is the original entropy

$\sum_{v=1}^V\dfrac{|D^v|}{|D|}\text{Ent}(D^v)$ is the weighed sum of all subsets

***We are finding the attribute that maximizes the information gain***

**case study** *(to be filled)*

**limitation of information gain**

![image-20220513193923372](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513193923372.png)

**other rules to calculate information gain**

1. gain ratio (used by C4.5)

![image-20220513194200510](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513194200510.png)

2. Gini index (used by CART)

![image-20220513194748772](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513194748772.png)

where $p_k$ and $p_k'$ are probabilities

**pruning of a tree**

1. overfitting of a tree

![image-20220513200219103](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513200219103.png)

![image-20220513200541170](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513200541170.png)

2. pruning

a. pre-pruning(预剪枝)

![image-20220513200742975](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513200742975.png)

case study *(to be filled)*

b. post-pruning(后剪枝)

**all of the non-leaf nodes can be pruned(including the root)**

![image-20220513201512696](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513201512696.png)

case study *(to be filled)*

**DT for continuous values**

continuous value discretization

example: bi-partition

![image-20220513202805571](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513202805571.png)

case study *(to be filled)*

**multivariant DT**

the efficiency of univariant DT is low on datasets like this:

![image-20220513203259588](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513203259588.png)

we can use a smooth curve as the decision boundary

case study *(to be filled)*

**random forest**

To reduce the variance of DT, we can **average different models**:

![image-20220513205210904](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513205210904.png)

*bagging*: used for models that is complex and easy to overfit

to train different models, we can use bootstrapping sampling:

![image-20220513205502038](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513205502038.png)

![image-20220513205523949](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513205523949.png)

*bagging for random forest*

![image-20220513210407326](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513210407326.png)

***out-of-bag's bag is not bagging's bag***

To predict, use majority voting:

![image-20220513210553912](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220513210553912.png)

## 8. Bayes Classifier

**Bayes decision theory**

basic idea:

![image-20220520192125175](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520192125175.png)

The loss function:

![image-20220520192230942](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520192230942.png)

$\lambda_{ij}$ is essentially the weight for each misclassified sample.

for a specific case:

![image-20220520192723004](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520192723004.png)

what it really does:

![image-20220520193023163](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520193023163.png)

Generative models:

![image-20220520193239254](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520193239254.png)

$P(c)$ is the frequency for class $c$ in the training set.

$P(x|c)$ is the class conditional probability, which we will do a good research

**Naïve Bayes Classifier**

Assumption: attributes are independent from each other

![image-20220520193603381](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520193603381.png)

and one more step, we can get the optimization object of NBC:

![image-20220520193713343](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520193713343.png)

Training the NBC:

1. estimate $P(c)$

It is just the frequency.

2. estimate $P(x_i|c)$

![image-20220520194010505](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520194010505.png)

![image-20220520194051328](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520194051328.png)

*Case Study*(to be filled)

What if an attribute does not appear？

![image-20220520195756599](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520195756599.png)

We can do Laplacian correction

![image-20220520195900820](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520195900820.png)

**Bayesian Network**

basic idea: not independent

![image-20220520200752122](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520200752122.png)

example of the network:

![image-20220520200927561](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520200927561.png)

Structure:

![image-20220520201008349](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520201008349.png)

example:
![image-20220520201040476](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520201040476.png)

Typical structures:

![image-20220520201237425](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520201237425.png)

moral graph(to be filled)

![image-20220520201434525](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220520201434525.png)

 General Steps

1. learning

2. inference(NP hard)

## 9. Neural Network & CNN

### a. NN

**artificial neuron**

![image-20220527192928560](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527192928560.png)

**activation function**

![image-20220527193234436](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527193234436.png)

**perceptron**

![image-20220527193310765](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527193310765.png)

**NN structure**

![image-20220527193401883](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527193401883.png)

**Deep NN**

![image-20220527193610521](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527193610521.png)

**How does this work**

a basic structure:

![image-20220527193837679](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527193837679.png)

Generally:

![image-20220527193939088](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527193939088.png)

As a classifier:

![image-20220527194208147](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527194208147.png)

*case study: handwritten digits recognition*

![image-20220527194739382](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527194739382.png)

**Backpropagation**

![image-20220527200138806](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527200138806.png)

*computation diagram*

![image-20220527200538028](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527200538028.png)

![image-20220527200849584](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527200849584.png)

a complicated example

![image-20220527201715684](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527201715684.png)

The steps of NN training is:

forward calculate the values

backwards calculate the derivative using **chain principle**.

![image-20220527203900266](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527203900266.png)

The model may be non-linear and **linear functions cannot fit non-linear function**.

### b. CNN

**Layers**

1. FCN(Fully Connected Layer)

![image-20220527205523512](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527205523512.png)

just like the tradition NN

2. Convolution Layer

![image-20220527205726715](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527205726715.png)

![image-20220527205848287](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527205848287.png)

an example of kernel:

![image-20220527210020106](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527210020106.png)

**The more similar, the more white, the bigger the conv value is**

**padding**

![image-20220527210741051](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527210741051.png)

![image-20220527210820029](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527210820029.png)

an example:

![image-20220527211224587](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527211224587.png)

![image-20220527211243753](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527211243753.png)

**summary**

![image-20220527211408852](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527211408852.png)

![image-20220527211619585](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527211619585.png)

3. Pooling Layer

![image-20220527212123322](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527212123322.png)

**Questions**

![image-20220527212324167](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220527212324167.png)

1. Pooling is FASTER
2. a) reduce the number of parameters; b) easy to learn features, flattening images maybe hard to learn features
