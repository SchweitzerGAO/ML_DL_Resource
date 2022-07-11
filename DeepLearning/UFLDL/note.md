# UFLDL note

*Note: This note mainly contains the DL part of this tutorial, also with something I didn't know about ML*

[TOC]

## 1. Softmax Regression

It is a generalization of logistic regression.

Here we assume that there are $K$ classes and given an input $x$ we are going to compute $P(y=k|x)$

the hypothesis function is:
$$
\begin{align}
h_\theta(x) =
\begin{bmatrix}
P(y = 1 | x; \theta) \\
P(y = 2 | x; \theta) \\
\vdots \\
P(y = K | x; \theta)
\end{bmatrix}
=
\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)^T} x) }}
\begin{bmatrix}
\exp(\theta^{(1)^T} x ) \\
\exp(\theta^{(2)^T} x ) \\
\vdots \\
\exp(\theta^{(K)^T} x ) \\
\end{bmatrix}
\end{align}
$$
the cost function is:
$$
\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}
$$
where $1\{\cdot\}$ is the indicator function ($1(\text {true})=1,1(\text {false})=0$)

the cost function of logistic regression is actually a special case for softmax regression:
$$
\begin{align}
J(\theta) &= - \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\
&= - \left[ \sum_{i=1}^{m} \sum_{k=0}^{1} 1\left\{y^{(i)} = k\right\} \log P(y^{(i)} = k | x^{(i)} ; \theta) \right]
\end{align}
$$
where
$$
P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{\exp(\theta^{(k)^T} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)^T} x^{(i)}) }
$$
**how it is generalized**

a mathematical property of softmax regression is that it has **redundant** set of parameters.

so we can substract a fixed vector $\psi$ from $\theta^{(j)}$ and $P$ remains unchanged:
$$
\begin{align}
P(y^{(i)} = k | x^{(i)} ; \theta)
&= \frac{\exp((\theta^{(k)}-\psi)^T x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^T x^{(i)})}  \\
&= \frac{\exp(\theta^{(k)^T} x^{(i)}) \exp(-\psi^T x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)^T} x^{(i)}) \exp(-\psi^T x^{(i)})} \\
&= \frac{\exp(\theta^{(k)^T} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)^T} x^{(i)})}.
\end{align}
$$
note that if we set $\psi=\theta^{(k)}$ we can still replace $\theta^{(k)}$ by $\mathbf 0$ without affecting the results

for logistic regression, we have the original $h_\theta(x)$ by definition:
$$
\begin{align}
h_\theta(x) &=

\frac{1}{ \exp(\theta^{(1)^T}x)  + \exp( \theta^{(2)^T} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)^T} x ) \\
\exp( \theta^{(2)^T} x )
\end{bmatrix}
\end{align}
$$
perform parameter subtraction:

we choose $\psi=\theta^{(2)}$ and we have:
$$
\begin{align}
h(x) &=

\dfrac{1}{ \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) + \exp(\vec{0}^\top x) }
\begin{bmatrix}
\exp( (\theta^{(1)}-\theta^{(2)})^\top x ) \\
\exp( \vec{0}^\top x ) \\
\end{bmatrix} \\

&=
\begin{bmatrix}
\dfrac{\exp( (\theta^{(1)}-\theta^{(2)})^\top x )}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\dfrac{1}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) }
\end{bmatrix} \\

&=
\begin{bmatrix}
 1 - \dfrac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)})  } \\
\dfrac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\end{bmatrix}
\end{align}
$$
replace $\theta^{(2)}-\theta^{(1)}$ by $\theta^\prime$ and we can see:

probability of one of the class: $\dfrac{1}{ 1  + \exp(- (\theta')^\top x^{(i)} ) }$  and the other one: $1 - \dfrac{1}{ 1 + \exp(- (\theta')^\top x^{(i)} ) }$

and we can get the cross-entropy function by $J=\mathbf yh(x)$ where $\mathbf y=\begin{bmatrix}1-y^{(i)}&y^{(i)}\end{bmatrix}$

**the gradient of the cost function**
$$
\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}
$$


