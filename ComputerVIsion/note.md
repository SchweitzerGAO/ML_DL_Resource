# CV note

[TOC]

## Lecture 2

### panorama stitching(全景拼接) I

Problem: How to stitch 2 images from 'similar' scenes together?

assumptions to stitch:

1. photos co-planar(共面) (not necessary)
2. undistorted(无畸变)

we can imply that the corresponding points $x_i$ and $x_i^{\prime}$  in the images are with the **same transform matrix** $H$ s.t. $x_i=Hx_i^{\prime}$

**steps**

1. find key points
2. construct descriptors for all key points
3. build correspondence for key points
4. solve $H$
5. apply $H^{-1}$ to image 2
6. stitch them together 

*linear mapping:*

$f(ax+by)=af(x)+bf(y)$

*affine mapping:*

$f(x)=Ax+b$

## Lecture 3

### linear transformation and some math

1. rotation: $x^{\prime}=\begin{bmatrix}R_{2\times2}\ \ \mathbf 0_{2\times1}\\ \mathbf0_{1\times2}\ \ 1\end{bmatrix}x$, where $R$ is orthogonal($R^T=R^{-1}$) and $\det(R)=1$, an example is $R=\begin{bmatrix}\cos\theta\ \ -\sin\theta \\  \sin \theta\ \ \cos\theta \end{bmatrix}$ **(constant length,dof=1)**
2. Euclidean: $ x^{\prime}=\begin{bmatrix}R_{2\times2}\ \ t_{2\times1}\\ \mathbf0_{1\times2}\ \ 1\end{bmatrix} x$, where $R$ is as shown above, $\vec t$ is translation matrix **(constant length,dof=3)**
3. similarity: $ x^{\prime}=\begin{bmatrix} sR_{2\times2}& t_{2\times1}\\ \mathbf0_{1\times2}\ &1\end{bmatrix}x$, where $R$ is as shown above, $s$ is a real number,$\vec t$ is translation matrix **(constant similarity ratio,dof=4)**
4. affine: $x^{\prime}=\begin{bmatrix}A_{2\times2}\ \ t_{2\times1}\\ \mathbf0_{1\times2}\ \ 1\end{bmatrix} x$, where $A$ is a non-singular matrix **(constant simple ratio,dof=6)**
5. projective: $x^{\prime}=Hx=\begin{bmatrix}a_{11}\ \ a_{12}\ \ a_{13} \\ a_{21}\ \ a_{22}\ \ a_{23} \\  a_{31}\ \ a_{32}\ \ a_{33} \end{bmatrix}x$, where $H$ is not singular **(constant cross ratio,dof=8)**

$x,x^{\prime}$ are homogenous coordinates

all the transformations above are **Lie Groups=manifold+group** -> **derivative exists**

**some references**

[simple ratio](https://baike.baidu.com/item/%E5%8D%95%E6%AF%94/7815415#2)

[cross ratio](https://baike.baidu.com/item/%E5%8D%95%E6%AF%94/7815415#2)

[homogenous coordinate](D:\COURSE_WORK_Bachelor\图形学2021秋\slides\第5章 图形变换与裁剪1.ppt)

[SVD](https://zhuanlan.zhihu.com/p/29846048)

[Lie Group](https://baike.baidu.com/item/%E6%9D%8E%E7%BE%A4/85603)

[differentiable manifold](https://baike.baidu.com/item/%E5%BE%AE%E5%88%86%E6%B5%81%E5%BD%A2/710877)

**PnP(Perspective-n-Point) problem(modified in lecture 10)**

Given the pixel coordinate and the world coordinate of $n$ 3D points, our aim is to estimate the **position** of the camera, that is to solve 

$$
\underset{R,t} {\text{argmin}}\ \sum_{i=1}^{n}\| u_i-\dfrac{1}{S_i}K[R,t] p_i\|_2^2
$$

where $u_i$ is the pixel coord of point set $P$, $p_i$ is the world coord of point set $P$, $R$ is the rotation matrix and $t$ is the translation matrix, $K$ is the matrix used to transform world coord to pixel coord(**intrinsics of the camera(相机内参)**), $S_i$ is the depth of the point $p_i$. We can use gradient to get the minimum because $R,t$ represents linear transformation and they are Lie Groups, [here](https://zhuanlan.zhihu.com/p/399140251) is more info about this problem

## Lecture 4

### panorama stitching II

**key points finding algorithm**

Harris Corner Detect:

*basic idea:*

![image-20220309083426072](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220309083426072.png)

*mathematics denotation & steps*

![image-20220309085415941](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220309085415941.png)

![image-20220309085347854](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220309085347854.png)

Note that from (1) to (2), **Taylor expansion of multi-variable function** is used, that is:

for $x,p\in\mathbb R^n$, $p$ is an infinitesimal, then $f(x+p)=f(x)+p^T\nabla f(x)+\dfrac {1}{2}p^T\nabla^2f(x)p+O(\|x\|^2)$ 

and we have:

![image-20220321140354624](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321140354624.png)

$S_w(\Delta x,\Delta y)$ forms an ellipse(proof can be found in HW1). **The shorter its axes are,the more likely the point is a corner point**

![image-20220321173420808](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321173420808.png)

$w_2$ has higher cornerness.

assume the eigenvalues of $M$ are $\lambda_1,\lambda_2\ (\lambda_1>\lambda_2)$ and the length of the semi-axes of the ellipse are $\dfrac{1}{\sqrt{\lambda_1}}$ (minor) and $\dfrac{1}{\sqrt{\lambda_2}}$(major) (proof can be found in HW1)

![image-20220321174126781](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321174126781.png)

so if a point is corner, $\lambda_1, \lambda_2$ should be both large

the measure is:

![image-20220321174325904](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220321174325904.png)

if $R$ is high, the point is prone to be a corner point

## Lecture 5

### panorama stitching III

**implementing Harris corner detection**

1. get the partial derivatives

![image-20220314081816571](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220314081816571.png)

note that $*$ is convlution operation(i.e. element-wise multiplication) (Prewitt operator)

2. follow the steps:

![image-20220314082647529](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220314082647529.png)

note that $g(x)$ is a Gaussian filter, whose functionality is to compute the weighted sum

Harris corner detection doesn't guarantee image scale invariance,reason:

![image-20220314084017941](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220314084017941.png)

**local descriptors of Harris corner**

![image-20220314084442364](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220314084442364.png)

There is a deficiency of such descriptors because:

- It is not rotation invariant
- It is not scale invariant

We need to move from points to regions(scale invariant regions,scale space)

**scale invariant point detection**

1. naive approach: exhaustive searching:

![image-20220316135941042](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316135941042.png)

2. automatic scale selection

what we want:

- the location of scale invariant point
- the **characteristic scales** of the point

like this:

![image-20220316141757790](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316141757790.png)

We need to find a function that is **single-peaked** and **covariant** with the scale of the image.$\text{LoG}$(Laplacian of Gaussian) exactly satisfies this.

Laplacian operator

![image-20220314095422163](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220314095422163.png)

actually, $\text{LoG}=\sigma^2(g_{xx}(\sigma)+g_{yy}(\sigma))$, which is often replaced by $\text{DoG}=g(x,y,k\sigma)-g(x,y,\sigma)$ in programming practice

Algorithm step:

1. choose $\sigma$'s as a list(usually 5?)
2. compute $\text{DoG}$ of the image for each $\sigma$
3. construct a 3D matrix like this:

![image-20220316144524130](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316144524130.png)

if the value of cross point is bigger than all of the 26 neighbours, then it is considered as a scale invariant point, $\sigma$ is considered as the characteristic scale 

## Lecture 6

### panorama stitching IV

**convolution**

![image-20220316083803604](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316083803604.png)

$I*h$:

- if $h$ is not reversed $180\degree$, the result is called **correlation**

- otherwise the result is **convolution**
  
  math definition

![image-20220316084723700](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316084723700.png)

$$
f(t)*g(t)=\int_{-\infin}^{+\infin}f(\tau)g(t-\tau)\text{d}\tau
$$

$g(t-\tau)$ is reversion, and the formula above explains why convlution is "multiple and add"

math properties(only after reversion):

- commutative
- associative

**Fourier transform**

time scope -> frequency scope

theorem:

![image-20220316090357730](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316090357730.png)

$f(t)*g(t)=\mathcal{F}^{-1}(F(u)G(u))$

where $\mathcal{F}$ is Fourier transformation, which is implemented through FFT algorithm in CUDA

**SIFT(Scale Invariant Feature Transform) descriptor**

basic steps:

![image-20220316092713192](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316092713192.png)

Assign keypoints orientations:

![image-20220316182437177](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316182437177.png)

descriptor construction steps:

![image-20220316191538269](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316191538269.png)

to visualize:

![image-20220316191904244](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316191904244.png)

what you can get from the pre-processed image through SIFT:

![image-20220316192257548](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220316192257548.png)

where $n$ is the number of scale invariant points

## Lecture 7

### panorama stitching V

**solve $H$(LSM)**

![image-20220323082928911](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323082928911.png)

at least 4 correspondence point pairs will determine a linear transformation $H$

that is because

![image-20220323083732052](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323083732052.png)

one point pair gives 2 equations,thus solves 2 dofs

to vectorize the equations above:

![image-20220323084757399](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323084757399.png)

**solution of linear equation group**

![image-20220323090409598](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323090409598.png)

![image-20220323090717260](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323090717260.png)

since only the ratios among the elements of $H$ take effect, we can set $h_{33}=1$

thus:

![image-20220323091235808](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323091235808.png)

**matrix differentiation**

1. vector function, scalar variable:

![image-20220323194535234](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323194535234.png)

2. matrix function, scalar variable

![image-20220323194718624](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323194718624.png)

3. scalar function vector variable

![image-20220323201015521](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323201015521.png)

4. scalar function matrix variable

![image-20220323201128105](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323201128105.png)

5. vector function vector variable

![image-20220323202642018](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323202642018.png)

OR

![image-20220323202704458](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323202704458.png)

some useful results:

1. ![image-20220323203011738](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323203011738.png)

2-7:

![image-20220323203048220](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323203048220.png)

8-10

![image-20220323203105126](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323203105126.png)

**proof of the results above**(to be filled, ensuring that all the proofs have been deducted by myself)

1. *Prove:*

![image-20220324161633630](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220324161633630.png)

2. *Prove:*

$\mathbf x^T\mathbf x=x_1^2+x_2^2+\dots+x_n^2$

$\dfrac{\text{d}\mathbf x^T\mathbf x}{\text{d}\mathbf x}=\begin{bmatrix}2x_1& 2x_2&\dots&2x_n\end{bmatrix}^T=2\mathbf x$

3. can be proven by the definitions of $\dfrac{\text{d}\mathbf y^T(\mathbf x)}{\text{d}\mathbf x}$ and $\dfrac{\text{d}\mathbf y(\mathbf x)}{\text{d}\mathbf x^T}$. See the definitions above.

4. 

*Prove:*

![image-20220324165039481](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220324165039481.png)

5. *Prove:*

we can get it immediately from (3) and (4)

6. *Prove:*

![image-20220325192932831](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220325192932831.png)

7. *Prove:*

![image-20220329230626062](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220329230626062.png)

8. *Prove:*

can be indicated from 7(just transpose $X$)

9. *Prove:*

![image-20220329232107578](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220329232107578.png)

**Lagrange multiplier**

find all possible extremum of a multi-variable function with constraints

![image-20220323203248206](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323203248206.png)

![image-20220323203310081](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220323203310081.png)

## Lecture 8

### panorama stitching VI

**LSM(Least Square Method)**

solve an over-determined equation group, i.e., the number of the equations in an equation group is more than that of the unknowns.

![image-20220328084148057](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328084148057.png)

![image-20220328085411504](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328085411504.png)

how it is solved

![image-20220328090209231](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328090209231.png)

$A^TA$ is invertible, the proof can be found in assignment 1

if the equation group is homogenous:

![image-20220328092259243](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328092259243.png)

![image-20220328092324727](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328092324727.png)

![image-20220328093100994](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220328093100994.png)

## Lecture 9

### panorama stitching VI

**RANSAC-based homography estimation**

RANSAC = RANdom SAmple Consensus(随机采样一致性)

algorithm:

![image-20220330085453682](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220330085453682.png)

for the question of homography estimation it is just like H-Net.

1. randomly choose 4 point pairs to estimate $H$
2. transform the image using $H$
3. calculate the distance of corresponding points of the projected image and the target image
4. perform (2)-(5) steps of the algorithm

example:

![image-20220406080517117](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406080517117.png)

**how to apply $H$ to the image**

bilinear interpolation

![image-20220330093659173](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220330093659173.png)

## Lecture 10

### measurement using a single camera(单目测量) I

 Problem: How to get the transformation matrix between the camera coord and the world coord?

**camera calibration**

What: solve the extrinsics and intrinsics of a camera

![image-20220406083215810](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406083215810.png)

1. distortion correction
2. object size measurement

![image-20220406090332464](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406090332464.png)

3. determine the location of the camera(PnP problem)

![image-20220406090404903](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406090404903.png)

**Why: build the correspondance of pixel CS and world CS**

How:

**pipeline**

1. model:

![image-20220406091742136](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406091742136.png)

2. how to calibrate:

use calibration board!

![image-20220406091821068](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406091821068.png)

3. required CS's

![image-20220406092211389](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406092211389.png)

4. pipeline:

a. world CS -> camera CS

![image-20220406094237798](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406094237798.png)

b. camera CS -> retinal CS

![image-20220406094830498](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406094830498.png)

homogenous form:

![image-20220406095503325](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220406095503325.png)

## Lecture 11

### measurement using a single camera II

c. camera CS -> normalized retinal CS

note: $f=1$ other parts are the same as that in camera CS -> retinal CS

![image-20220411081403903](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411081403903.png)

d. retinal CS -> pixel CS

![image-20220411081937371](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411081937371.png)

where:

$dx$ is the length of one pixel on $x$ axis

$dy$ is the length of one pixel on $y$ axis

$(c_x,c_y)$ is the optical center(principle point)

 if axes $u$ and $v$ are not perpendicular:

![image-20220411082654886](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411082654886.png)

e. putting them all together:

![image-20220411084346661](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411084346661.png)

*some notes on MATLAB and OpenCV camera calibration*

![image-20220411085442369](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411085442369.png)

We will not consider $s$ in next lectures

f. take normalized retinal CS into consideration:

![image-20220411085716072](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411085716072.png)

g. take distortion into consideration

*modelling two kinds of distortions*

![image-20220411090418237](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411090418237.png)

note that when we add both the distortions together, we just consider one of the two $x_n$'s

*If the camera is fisheye camera*

i.e., the FOV is very large(larger than $100\degree$):

![image-20220411090946183](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411090946183.png)

We have to use another model:

![image-20220411091040880](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411091040880.png)

In this lecture, we just consider the first one model:

![image-20220411091351532](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411091351532.png)

As the distortion can't be represented as matrix multiplication, it is represented as an operator $\mathcal D$ 

5. algorithm framework

![image-20220411093207377](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411093207377.png)

(1) calibration board

![image-20220411094720068](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220411094720068.png)

## Lecture 12

### measurement using a single camera III

(2) get some images of the calibration board

![image-20220413082829461](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220413082829461.png)

(3) solve this optimization equation

![image-20220413082954073](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220413082954073.png)

This optimization objective illustrates the **sum of Euclidean distance between the pixel coord computed by eq.(8) and the observed pixel coord.**

the amount of unknowns in the objective is $9+6M$

To solve eq.(9) we have to find a form to **uniquelly represent a rotation by only 3 numbers**

**the axis-angle**

![image-20220413090353838](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220413090353838.png)

transformation between rotation matrix and axis-angle:

![image-20220413090828524](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220413090828524.png)

Altogether, we have $2\times M\times N$ equations and $9+6M$ unknowns

As this problem is **not convex**, we have to have a good estimate of the initial values of the parameters

## Lecture 13

### projective geometry fundamental I

**vector operations** 

representation, length(L2-norm) and normalize

![image-20220420080406428](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420080406428.png)

2. dot product(scalar product)

![image-20220420080515062](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420080515062.png)

3. cross product(vector product,direction is vertical to the plane of $\mathbf a$ and $\mathbf b$)

![image-20220420080605439](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420080605439.png)

the length of the result is $|\mathbf a||\mathbf b|\sin\theta$, where $\theta$ is the angle of $\mathbf a$ and $\mathbf b$

properties

![image-20220420081332182](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420081332182.png)

a new definition of cross product

![image-20220420081421805](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420081421805.png)

3. mixed product

![image-20220420081748949](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420081748949.png)

the geometric representation of mixed product: **the volume of a box**

![image-20220420081839674](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420081839674.png)

properties:

![image-20220420082214550](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420082214550.png)

![image-20220420082330110](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420082330110.png)

**Fundamentals of projective geometry**

1. homogenous coordinate

*for a normal point*

![image-20220420084341697](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420084341697.png)

converting from homogenous to inhomogenous:

![image-20220420084452725](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420084452725.png)

geometric interpretion:

![image-20220420085248145](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420085248145.png)

![image-20220420085638023](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420085638023.png)

builds correspondence between a line and a point in a plane

if the plane is parallel to the plane:

**definition of infinity point**

![image-20220420085915777](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420085915777.png)

**Infinity line**

![image-20220420090643754](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420090643754.png)

the 'crossing line' of $O\mathbf{e_1e_2}$ and $\pi_0$

**projective plane**

properties:

![image-20220420091115013](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420091115013.png)

![image-20220420091200079](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420091200079.png)

note that there is **only 1 cross point** between parallel lines, it is from the definition like this(a ring):

![image-20220420091608106](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420091608106.png)

**lines in homogenous coordinate:**

![image-20220420092126001](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420092126001.png)

*coplanar triple vectors=> mixed product=0*

![image-20220420092410421](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420092410421.png)

![image-20220420092531256](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420092531256.png)

![image-20220420092655543](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420092655543.png)

**determine the cross point **

![image-20220420093134339](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420093134339.png)

![image-20220420093216018](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220420093216018.png)

## Lecture 14

### projective geometry fundamental II

**converting inhomogenous line to homogenous line**

![image-20220425081357869](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425081357869.png)

a. represent $x$ and $y$ with $\dfrac{x}{z}$ and $\dfrac{y}{z}$

b. multiply $z$ at both sides of the equation and that is homogenous equation of the line

**duality theorem**

![image-20220425081556145](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425081556145.png)

**more results**

![image-20220425082639211](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425082639211.png)

![image-20220425082737236](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425082737236.png)

**examples of projective transformation**

![image-20220425082918098](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425082918098.png)

### measurement using a single camera IV

**rough estimation of calibration parameters(intrinsic)**

at this point we do not consider distortion and set all the initial value of parameters related with distortion to 0

![image-20220425084357941](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425084357941.png)

the $z$ coord of $\mathbf P$ is 0

*result 1*

![image-20220425092543620](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425092543620.png)

*result 2*

![image-20220425092559903](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425092559903.png)

just use the definition of inner product.

**vanishing points**

![image-20220425093100480](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425093100480.png)

example:

![image-20220425095402869](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220425095402869.png)

## Lecture 15

### measurement using a single camera V

**examples of vanishing points**

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427081642900.png" alt="image-20220427081642900" style="zoom:50%;" />

<img src="C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427081709840.png" alt="image-20220427081709840" style="zoom:50%;" />

**properties of vanishing points**

![image-20220427082033537](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427082033537.png)

*Result 3*

![image-20220427082254774](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427082254774.png)

***Result 4 (key result!)***

![image-20220427082942449](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427082942449.png)

using result 4, we can get this($H$ is the homography matrix **from WCS to the pixel CS**):

![image-20220427085954072](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427085954072.png)

if we have $M$ images of calibration board,we can get $2M$ equations like below:

![image-20220427090447588](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427090447588.png)

Then we can solve $K$

**OpenCV implementation of initial estimation of intrinsics**

OpenCV applies a simplified approach to estimate the intrinsics of the camera, it assumes $(c_x,c_y)$ to be $(w/2,h/2)$ where $(w,h)$ is the width and height of the image.

OpenCV estimates the intrinsics as below:

1. decomposite $K$ with $PQ$:

![image-20220427102333785](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427102333785.png)

2. rewrite equation (11) :

![image-20220427102944059](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427102944059.png)

![image-20220427103433797](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427103433797.png)

3. solve it!

![image-20220427103450354](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427103450354.png)

**initial estimation of extrinsics**

![image-20220427092911469](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427092911469.png)

$H$ is the homography **from WCS to the normalized retinal CS**

![image-20220427093359969](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220427093359969.png)

## Lecture 16

### measurement using a single camera VI

**non-linear LS**

![image-20220509080759416](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509080759416.png)

### non-linear least square I

**what will it cover?**

![image-20220509081939840](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509081939840.png)

**basic concepts**

1. convex function

![image-20220509082234176](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509082234176.png)

![image-20220509082331784](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509082331784.png)

2. local minimizer

![image-20220509082903645](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509082903645.png)

3. Taylor expansion, gradient and Hessian matrix

![image-20220509083121039](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509083121039.png)

4. stationary point

![image-20220509083342870](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509083342870.png)

5. sufficient condition for a local optima

![image-20220509083733064](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509083733064.png)

**descent methods**

1. guarantee

![image-20220509084510983](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509084510983.png)

using 1-order info: gradient

2-order: Hessian

2. descent direction

![image-20220509085140151](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509085140151.png)

3. descent methods

![image-20220509085407907](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509085407907.png)

4. 2-phase method

*general framework of algorithm*

![image-20220509085944754](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509085944754.png)

note:

![image-20220509090100801](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509090100801.png)

*gradient descent*

![image-20220509091534278](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509091534278.png)

*Newton's method*

compute the descent direction **near convergence**

![image-20220509092947517](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509092947517.png)

![image-20220509093242338](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509093242338.png)

properties & *hybrid method*:
![image-20220509093609442](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220509093609442.png)

## Lecture 17

### non-linear least square II

*line search*

![image-20220511081715186](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511081715186.png)

principles to choose $\alpha$

![image-20220511081932616](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511081932616.png)

Why doesn't ML(DL) adopt line search as a method to choose $\alpha$? 

**The scale of the problem may be enumerous! The cost is too high to calculate the gradient**

5. 1-phase method

*general idea*

![image-20220511083214601](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511083214601.png)

theorem: if the 2-order partial derivative of a function is semi-positive(exists), then the function is convex.

*trust region method*

![image-20220511085950805](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511085950805.png)

example:

![image-20220511085732133](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511085732133.png)

how to update $\Delta$

evaluation: gain ratio

![image-20220511090531773](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511090531773.png)

update $\Delta$ by gain ratio

![image-20220511091020694](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511091020694.png)

*damped method*

In this method, the step is determined as (the idea is like regularization term in ML):

![image-20220511091749584](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511091749584.png)

to solve:

![image-20220511091903335](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511091903335.png)

note:

![image-20220511091930367](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511091930367.png)

algorithm:

![image-20220511092147227](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511092147227.png)

how to update $\mu$

still by gain ratio $\rho$ !

![image-20220511092458933](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511092458933.png)

*damped Newton method*

set $\mathbf c$ to the gradient of $F$ and $\mathbf B$ to the Hessian of $F$

![image-20220511092814655](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511092814655.png)

**Non-linear least square problem**

**Basic Concepts**

*Formulation*

![image-20220511093238358](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220511093238358.png)

## Lecture 18

### non-linear least square III

Recall: linear LS

![image-20220518081634191](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518081634191.png)

**Basic Concepts**

*Jacobian Matrix*

![image-20220518082036067](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518082036067.png)

gradient + transpose = Jacobian

for the basic formulation $F(\mathbf x)$ of the non-linear LS, we have:

![image-20220518082721012](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518082721012.png)

to make this easier:

![image-20220518082824007](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518082824007.png)

We can rewrite this to the form with Jacobian matrix.

For 2-order:

![image-20220518083230099](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518083230099.png)

**Gaussian-Newton Method**

![image-20220518083635373](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518083635373.png)

1. Taylor expansion
2. gradient = 0
3. get the approximation analytical solution of $L$

To compare it with the general model mentioned in the 1-phase method:

![image-20220518085252802](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518085252802.png) 

Note that $J^TJ$ is **not Hessian matrix**

some notes:

![image-20220518085756261](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518085756261.png)

**Levenberg-Maquardt Method**

Can be regarded as damped G-N method.

![image-20220518090017102](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518090017102.png)

differences between LM and GN methods:

1. LM doesn't require $J$ is column full-rank
2. LM has a penalty item $\dfrac{1}{2}\mu\mathbf h^T\mathbf h$

*Prove $J^TJ+\mu I$ is positive-definite*

![image-20220518090651396](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518090651396.png)

*updating strategy for $\mu$*

![image-20220518091711213](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518091711213.png)

*stopping critetia*

![image-20220518092021930](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518092021930.png)

*the pseudo-code for LM method*

![image-20220518092217071](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220518092217071.png)

**Dog-leg Method**

*To be filled*

## Lecture 19

### measurement using a single camera VII

using L-M method to solve the non-linear least square:

![image-20220523081733774](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523081733774.png)

for $2MN$: expand each vector, we can get 2 equations

for $9+6M$: refer to the note before

The core problem is to **compute the gradient**:

*How to compute the gradient?*

![image-20220523082724823](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523082724823.png)

for $\mathbf d_i^T$: the axis-angle formuation of rotation(refer to the note above)

some denotations:

![image-20220523083501982](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523083501982.png)

results:

1.

![image-20220523084139924](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523084139924.png)

2.

![image-20220523084439994](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523084439994.png)

note that $\mathbf p$ is not directly relevant with $\mathbf k^T$, we should use the chain principle of derivative. Then we can get 2

for 3: refer to the assignment problem 1

3. ![image-20220523084832010](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523084832010.png)

4.

![image-20220523085807632](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523085807632.png)

for 1: refer to assignment problem 2

After calibration, we can directly do something using the intrinsic, the distortion coeffs and the extrinsic

**1. undistortion**

![image-20220523090943818](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523090943818.png)

**2. BEV(Bird's Eye View) generation**

To generate BEV, we need 3 coordinate systems

![image-20220523091143330](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523091143330.png)

basic idea:

![image-20220523091714136](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523091714136.png)

The key problem is how to get $P_{W\to I}$ and $P_{B\to W}$

for $P_{B\to W}$

1.

![image-20220523092241376](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523092241376.png)

2.

![image-20220523092301123](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523092301123.png)

for $P_{W\to I}$

![image-20220523092410324](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523092410324.png)

an example:

1. undistortion

![image-20220523092535868](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523092535868.png)

2. bird's eye view generation

![image-20220523092658180](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220523092658180.png)

## Lecture 20

### Machine Learning Basics & CNN I

**basics**

The variable of P-R curve is **the threshold of classification**:

![image-20220525082049374](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525082049374.png)

***different curves correspond to different hyper-parameters***

Handling skewed data:

![image-20220525083435169](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525083435169.png)

an oversampling algorithm: SMOTE

![image-20220525083613552](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525083613552.png)

**softmax regression**

1. softmax operation

![image-20220525090730221](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525090730221.png)

2. cross entropy

![image-20220525090856599](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525090856599.png)

cross entropy can be an instance for $dist$

How to get this?

![image-20220525091442544](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525091442544.png)

Information amount:

![image-20220525092147432](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525092147432.png)

it is relevant with the probability that an event take place, actually, $p(x)$'s increment results in $h(x)$'s decrement. 

![image-20220525092623594](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525092623594.png)

if the base of log is 2, the unit of entropy is bit, else if the base is $e$ , the unit is nat

an example:

![image-20220525093038686](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525093038686.png)

**Neural Network**

back-propagation

**CNN**

some terminologies

![image-20220525093412857](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525093412857.png)

## Lecture 21

### Machine Learning Basics & CNN II

some terminologies:

![image-20220525093412857](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220525093412857.png)

column 1: tricks of DL

column 2: backbones of DL

column 3: object detection algorithms

column 4: DL frameworks

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-01-08-33-34-image.png)

column 1: GPU types

column 2: datasets

column 3.1: problems of DL

1. Gradient Vanishing: with the increment of the depth of the network, the gradient to be updated becomes 0 **(almost resolved)**.

2. Adversarial Samples(对抗样本): add some noise to the image which doesn't affect human recognition but affect much to machine recognition **(not resolved)**

column 3.2: famous scientists of DL

column 4:  virtual versions of column 2.

### Applications of CNN: automatic parking

*to be filled*

## Lecture 22

### Introduction of Numerical Geometry

how to measure the similarity of 3D models?

**Shapes vs Images**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-16-07-image.png)

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-22-29-image.png)

non-rigid transformation shown above is projected to another space where it is easier to measure the similarity only by projective transformations

**Basic Concepts**

1. distance

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-30-44-image.png)

2. metric

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-32-04-image.png)

3. metric balls

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-32-54-image.png)

4. Homeomorphisms (同胚) 

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-47-44-image.png)

an interesting example:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-48-45-image.png)

5. Isometries(保距)

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-49-47-image.png)

- Euclidian Isometry

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-50-45-image.png)

- Geodesic Isometry

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-51-50-image.png)

**Metric for discrete geometry**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-56-16-image.png)

FMM algo. is a continuous version of Dijkstra algorithm

**Sampling for 3D models**

$r$-covering:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-08-59-47-image.png)

$r^\prime$-separated:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-09-01-36-image.png)

a good sampling of a 3D model is to choose a point set that is r-covering and r-separated

a simple algorithm to satisfy this requirement is called **farthest point sampling**:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-09-04-59-image.png)

****

**Voronoi decomposition**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-09-19-05-image.png)

decrease the time complexity of certain algorithm

**Delaunay tessellation(三角化)**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-06-09-31-49-image.png)

## Lecture 23

### Rigid shape analysis

**Euclidean isometries removal**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-12-53-image.png)

How to remove this?

1. find the centroid:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-15-28-image.png)

2. find the axes

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-16-39-image.png)

It is essentially PCA

**ICP Algorithm**

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-24-41-image.png)

Problem definition:

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-35-35-image.png)

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-36-45-image.png)

### Review

**Panorama Stitching**

1. Harris corner detection

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-44-35-image.png)

2. Scale Invariant Point Detection

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-52-01-image.png)

3. SIFT

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-08-56-24-image.png)

4. RANSAC

**Projective Geometry**

1. Infinity point

**Non-linear LS Optimization**

1. Methods

L-M Method!

**Single Camera Measurement**

1. Camera Calibration

![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-09-11-42-image.png)![](C:\Users\CharlesGao\AppData\Roaming\marktext\images\2022-06-08-09-14-56-image.png)

2. BEV generation

**ML & DL**
