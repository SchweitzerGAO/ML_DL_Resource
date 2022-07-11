# Note of Papers: ML Project

[TOC]

## 1. SPV-SSD

### a. introduction

problem to solve: real-time 3D object detection in self-driving

current methods: monocular/stereo camera: **no depth information**

LiDAR(Light Detection And Ranging) can obtain accurate depth information but **unfriendly to small objects**

1. PointPainting and its drawbacks

Opportunities to combine the 2 methods together, but the **dilemma lies in the fact that BEV is difficult to align with the front view of the camera**

A new style in the input stage can solve this problem, among which is PointPainting.

PointPainting projects the point cloud into the image semantic segmentation results and appends the semantic class scores to each point. It helps to detect small objects by semantic information but decreases the precision of large objects.

![image-20220428153911515](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220428153911515.png)

The reason is that the sequential fusion makes subsequent detection network heavily rely on the quality of semantic segmentation *(how and why?)*

![image-20220428154236686](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220428154236686.png)

2. current 3D detector & database sampling and its drawbacks

**current 3D detector**

destroys the hidden spatial distribution information *(depth?)* occludes everything behind the detected object

With the visibility constraint, it can estimate the free area distribution in the 3D space and can provide context information for object detection *(why?)*

**database sampling**

paste virtual objects into point cloud, often behind walls or buildings, ignoring the visibility and authenticity

Author adopted visibility feature to assist both object detection and data augmentation

3. Existing voxel-based detectors and its drawbacks

anchor-based detection head, **expensive in time**

Author used anchor-free 

4. Works of authors

a. supervised-PointRendering

b. fuse with point clouds and image semantic features

c.  anchor-free detection head powered by Sim-OTA label assignment

### b. related work

1. 3D object detection with point cloud

*Single-stage:*

Voxel-Net: voxelize point cloud and extract feature by a tiny PointNet

SECOND: sparse convolution + submanifold sparse convolution; accelerate

Point-GNN: novel GNN; comparable performance with 2-stage SOTA models

SASSD: auxiliary network for box center regression & segmentation

*2-stage:*

Voxel-RCNN: generates ROI

PV-RCNN: leverage the advantage of 3D voxel-CNN and PN-based set abstraction to learn more features

Part-A2: enrich ROI by predicting intra-object part locations; reduce the ambiguity of bounding boxes.

2. multi-modal fusion

*object centric fusion*

MV3D&AVOD: extract ROI; deep feature fusion; loses spatial info 

*continuous feature fusion*

ContFuse: mapping the point cloud to the image plane; fuzzy feature align

*detection seeding*

Frustum PointNet & ConvNet: limit the frustum search space to seed the 3D proposal; rely heavily on 2D detector performance; low recall

*sequential fusion*

LRPD & PointPainting: both use the output of image semantic segmentation network to assist object detection; PointPainting projects LiDAR points into the semantic segmentation results and feed the painted points to 3D object detector; 'boundary blurring effect'

3. visibility representation

*visibility representation*

2D probabilistic occupancy map based on sonar readings to navigate the mobile robots

general 3D occupancy map to describe the space state, indicating the occupied, free and unknown area

core: raycasting algorithm

*in 3D object detection*

integrate the occupancy grid map into the probabilistic framework to detect objects with known surface

reconstruct the spatial visibility state through raycasting algorithm and convert 3D spatial features to 2D multichannel feature maps, and then integrate the 2D feature maps into the PointPillar, can be directly concatenated with voxelized point cloud; better data alignment

### c. Proposed Method

1. backbone

![image-20220505111029283](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220505111029283.png)

2. supervised PointRendering

backbone part 4?

![image-20220506110136235](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506110136235.png)

focal loss

![image-20220506113335660](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506113335660.png)

3. extraction of spatial visibility feature

ray-casting algorithm

![image-20220506154115286](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506154115286.png)

![image-20220506162538321](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506162538321.png)

light blue(green): unknown

dark blue(purple): free

red(small dots): occupied

4. anchor-free detection head with improved label assignment

![image-20220506164028135](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506164028135.png)

5.  loss function

a. overall loss $L$

![image-20220506222009493](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220506222009493.png)

$L_{\text{head}}$: loss of detection head

$L_{\text{seg}}$: loss of foreground segmentation

$L_{\text{ctr}}$: loss of center regression

$\omega=0.9$; $\mu=2.0$ (Hyper-parameters)



b. detection head loss $L_{\text{head}}$

![image-20220507162445652](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220507162445652.png)

c. center regression loss $L_{\text{ctr}}$

*Where is $L_{\text{seg}}$? focal loss?*

 

## 2. SECOND(for convolution)

**work**

- We apply **sparse convolution** in LiDAR-based object detection, thereby greatly increasing the speeds of training and inference.  
- We propose an **improved** method of sparse convolution that allows it to **run faster**. 
- We propose a novel **angle loss regression** approach that demonstrates better orientation regression performance than other methods do.
- We introduce **a novel data augmentation method for LiDAR-only learning problems** that greatly increases the convergence speed and performance

**approach**

![image-20220517160548919](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220517160548919.png)

1. point cloud grouping

   ![image-20220519215103323](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519215103323.png)

2. VFE(Voxelwise Feature Extractor)

   ![image-20220519215700249](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519215700249.png)

3. Sparse Convolutional Middle Extractor

   ![image-20220519220822780](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519220822780.png)

   ![image-20220519220934420](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519220934420.png)

   ![image-20220519221548361](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519221548361.png)

   instead of converting 3 to 4, the author used 5 to avoid computing 0s in the sparse convolutional net. 

   ![image-20220519222006312](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519222006312.png)

   

   where $R_{k,j}$ is called **Rule**

   ![image-20220519222051766](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519222051766.png)

   ![image-20220519223034196](C:\Users\CharlesGao\AppData\Roaming\Typora\typora-user-images\image-20220519223034196.png)

   

4. RPN(Region Proposal Network)

   

## 3. CenterNet(for anchor-free detection)



## 4. SA-SSD



## 5. CIA-SSD
