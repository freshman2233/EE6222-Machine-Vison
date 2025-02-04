# EE6222 Machine Vison

# 1.**Course Aims**

This course aims to introduce to students the basic concepts of vision-based automation systems in industrial and practical settings. Vision-based automation system involve image processing and analysis, video analysis, three-dimensional data processing, machine learning and intelligence. This course covers these topics appropriately.

# 2.**Intended Learning Outcomes (ILO)**

By the end of this course, students would be able to:

1. Understand the basic concepts of image pre-processing & analysis, feature extraction and pattern classification.

2. Understand the basic concepts of three-dimensional image and video analysis and recognition.

3. Apply the machine vision concepts to develop simple automation systems.
4. The students should be able to understand fundamental theories and algorithms in machine learning. Understand deep neural networks and know how to leverage them in solving complex computer vision problems in automation systems.

# 3.**Course Content**

Machine Vision plays a major role in automation. This course is centered around the usage of images, videos and other visual information for automation. Hence, this course covers image processing, image analysis, image recognition, machine learning, video recognition, three dimensional machine vision and their applications in automation.

# 4.**Course Outline**

**Fundamentals of Image, Processing and Transforms (6 hours)**

Image formation. Pre-processing. Spatial domain & frequency domain operations.

**Image Denoising, Enhancement and Manipulation (6 hours)**

Nonlinear processing and Histogram equalization. Rank order statistical filter, Binary and gray scale morphological operations. Local orientation of image.

**Decision, Classification and Machine Learning (9 hours)**

Decision boundaries. Machine learning for decision. Unsupervised learning and K-means clustering. Nearest neighbor classifiers, Linear classifiers, Minimum distance classifiers, Neural networks, Deep convolutional neural networks and their application in machine vision.

**Visual Data Dimensionality Reduction (6 hours)**

Data representation, Eigenvalue and eigenvectoer decomposition, principal component analysis, Linear discriminant analysis, critical roles of dimensionality reduction in visual data recognition.

**Three-dimensional Machine Vision (6 hours)**

Reflectance map, shape from shading, Stereovision techniques. Motion detection, Surface recovery from range data. Binocular stereo vision. Model-based recognition. 3D model representations.

**Video Recognition (6 hours)**

Video representation, Spatio-temporal feature extraction from video, Gesture recognition, Action recognition and activity recognition.

# 5. **Reading and References**

1. Davies E. R., Computer Vision: Principles, Algorithms, Applications, Learning, Elsevier Science, Academic Press, 2017.

2. Rafael C. Gonzalez, Richard E. Woods, Digital image processing, Pearson, 4th Edition, 2018

3. Duda R. O., Hart P. E. , and Stork D. G., Pattern Classification, John Wiley & Sons, 2001



# 6. Machine vision, exam scope

Chapter 2, linear translation invariant systems and transformations

In Chapter 9, dimension reduction of visual data is used as feature extraction

Chapter 10, Neural Networks and Deep Learning, from MLP to CNN

Chapter 12, Video analysis

Chapter 13, Video recognition

Chapter 14, 3D machine perception

Chapter 15, 3D machine Vision

第二章，线性平移不变系统与转换

第九章，视觉数据降维作为特征提取

第十章，神经网络与深度学习，从MLP到CNN

第十二章，视频分析

第十三章，视频识别

第十四章，三维机器感知

第十五章，三维机器视觉

# 7. Details

## 7.1 English

2 LSI linear translation invariant systems and transformations

2.1 Image decomposition and linear translation invariant image processing system

2.1.1. Image decomposition

2.1.2. Pulse

2.1.3. Translation and zooming pulse

2.1.4. Image decomposition



2.2 Two-dimensional convolution and its properties

2.2.1. Processing system

2.2.2. Linear processing system

2.2.3 Two-dimensional convolution

2.2.4 Convolution characteristics

2.2.5 Impulse response





2.3 Two-dimensional Fourier transform and its properties

2.3.1 Fourier transform and its inverse transformation

2.3.2 Independent and separable

2.3.3 Discrete Fourier Transform (DFT)

2.3.4 Discrete Fourier Transform (DFT) properties

2.3.4.1 Can be decomposed

2.3.4.2 Periodic, symmetric, linear, Convolution

2.3.4.3 Translation, rotation and translation do not change



2.4 Image Sampling

2.4.1 The image becomes discrete after continuous sampling

2.4.2 Sampling function

2.4.3 Image x sampling

2.4.4 Restoring Continuous Images

2.4.5 Sampling theorem



9 Visual data dimensionality reduction as feature extraction

9.1 Feature Extraction and Dimension Reduction

9.2 PCA, Principal component analysis

9.3 LDA, linear discriminant analysis

9.4 Classification of subspace/feature space



10 Neural Networks and Deep Learning, from MLP to CNN

10.1 Network structure and neuron model

10.1.1 Neural networks and deep CNNS

10.1.2 Neuron model

10.1.3 Activating functions



10.2 Multi-layer perceptron and backpropagation

10.2.1 ANN

10.2.1.1 Multi-Layer Perceptron (MLP)

10.2.1.2 Single-layer neural network

10.2.1.3 Learning curve and decision boundaries

10.2.1.4 Backpropagation

10.2.1.5 Local Minimum Problem

10.2.1.6 Understanding Overfitting

10.2.1.7 Conclusion of neural network



10.2.2 History of neural networks

10.2.3 Function of image recognition module

10.2.4 Supervised learning

10.2.5 Deep Learning

10.2.6 Problems with machine learning

10.2.7 Regularization



10.3 Convolutional Neural network CNN

10.3.1 Convolutional network CNN seems to be different from MLP

10.3.2 Features of Convolutional network CNN

10.3.3 1x1 Convolution

10.3.4 Advantages of Convolutional Networks, CNN







12 Video: Generated

12.1. Autoencoder



12.2. Antagonistic learning

12.2.1. GANs

12.2.2.Conditional GANs

12.2.3.Super Resolution GANs

12.2.4. CycleGAN

12.2.5.Diffusion Probabilistic Models

12.2.6.Video GAN

12.2.7.StoryGAN: A sequential condition GAN for story visualization

12.2.8. Two-stream VAN for video generation

12.2.9.Sora: Video generation model as a world simulator



13 Video: Analysis/Enhancement

13.1. Target detection and tracking

13.1.1 Detection and tracking

13.1.1.1 Detecting/dividing objects

13.1.1.2 Association Detection



13.1.2 Identifying the Re-ID again



13.1.3 Joint Detection and Embedding (JDE)







13.2. Behavior recognition

13.3. Video event/exception detection

13.3.1 Behavior Identification

13.3.2 Long-term Recurrent Convolutional Network Long-term Recurrent Convolutional Network

13.3.3 C3D: 3D convolutional network

13.3.4 Hidden Two streams



13.4. Video enhancement

13.4.1 Blur model

13.4.2 Deconvolution of fuzzy images

13.4.2.1 Point Spread Function (PSF) Point Spread Function (PSF)

13.4.3 Deep video deblur

13.4.4 Video Turbulence Effect Remove. Video turbulence effect remove



13.5. Optical flow

13.5.1 What is Optical Flow?

13.5.2 Optical Flow Hypothesis

13.5.2.1 Spatial coherence

13.5.2.2 Persistence of time

13.5.2.3 Constraints on the brightness constant

13.5.3 Brightness constant equation

13.5.4 Constraints on the brightness constant



13.6. Video segmentation

13.6.1. Introduction to segmentation: pixel input, pixel output

13.6.2 Convnets are classified

13.6.3 R-CNN

13.6.4 U-Net Architecture

13.6.4.1 Shrink phase

13.6.4.2 Expansion Phase

13.6.4.3 U-Net Summary

13.6.5 Image segmentation and video segmentation





14 3D

14.1 3D Method

14.1.1. Flight time

14.1.2. Lidar

14.1.3. Structured light

14.1.4. Motion construction

14.1.4.1 Introduction: Reconstruct scene geometry and camera position from two or more images

14.1.4.2 Pixel mapping: Tracking

Harris Corner

SIFT

Scale space

Descriptor

Object Detection

SuperPoint

Image robust matching strategy

(a) Key point space outlier rejection

(b) Match many features - find a good identical graph

RANSAC cycle

SuperGlue

LoFTR: Transformer-based local feature matching without detector

Key point tracking: Lucas-Kanade Tracker

affine cameras



14.1.4.3 Projection model

Affine structure in motion affine structure

Affine ambiguity. Affine ambiguity





14.1.5. Stereoscopic imaging

14.1.5.1.3D Video Principles

14.1.5.2. Stereo depth

14.1.5.3. Stereo matching



14.2 3D meets deep learning

14.2.1 Introduction to 3D

14.2.2 3D CNN volume data



14.3 GAN for 3D

14.3.1 From single image to volume

14.3.2 From single image to point cloud

14.3.3 From image to shape

14.3.4 From image to surface

## 7.2 Chinese

2 LSI线性平移不变系统与转换

2.1 图像分解与线性平移不变图像处理系统 

2.1.1.图像分解

2.1.2.脉冲 

2.1.3.平移和放大缩小脉冲

2.1.4.图像分解 





2.2 二维卷积及其性质 

2.2.1.处理系统 

2.2.2.线性处理系统

2.2.3 二维卷积 

2.2.4 卷积特性 

2.2.5 脉冲响应 





2.3 二维傅里叶变换及其性质 

2.3.1 傅里叶变换及其逆变换

2.3.2 独立可分离

2.3.3 离散傅里叶变换(DFT) 

2.3.4 离散傅里叶变换(DFT) 性质

2.3.4.1可分解

2.3.4.2 周期、对称、线性、卷积

2.3.4.3 平移、旋转和平移不变变换



2.4 图像采样 

2.4.1 连续采样后变为离散图像

2.4.2 采样函数

2.4.3 图像x采样

2.4.4 恢复连续图像

2.4.5 采样定理



9 视觉数据降维作为特征提取

9.1 特征提取/降维介绍

9.2 PCA，主成分分析

9.3 LDA，线性判别分析

9.4 子空间/特征空间的分类



10 神经网络与深度学习，从MLP到CNN

10.1网络结构和神经元模型

10.1.1 神经网络和深度CNN 

10.1.2 神经元模型 

10.1.3 激活函数



10.2多层感知器与反向传播

10.2.1 人工神经网络 ANN

10.2.1.1 多层感知器(MLP) 

10.2.1.2 单层神经网络 

10.2.1.3 学习曲线和决策边界 

10.2.1.4 反向传播  

10.2.1.5 局部极小问题 

10.2.1.6 理解过拟合 

10.2.1.7 神经网络的结论 



10.2.2 神经网络历史

10.2.3 图像识别模块的功能 

10.2.4 监督学习 

10.2.5 深度学习

10.2.6 机器学习的问题

10.2.7 正则化 



10.3卷积神经网络CNN

10.3.1 卷积网络CNN似乎与MLP不同

10.3.2 卷积网络CNN的特点 

10.3.3 1x1卷积

10.3.4 卷积网络的优点，CNN 







12 视频：生成

12.1.自动编码器



12.2.对抗学习

12.2.1.GANs

12.2.2.Conditional GANs

12.2.3.Super Resolution GANs

12.2.4.CycleGAN

12.2.5.Diffusion Probabilistic Models 

12.2.6.Video GAN

12.2.7.StoryGAN：用于故事可视化的顺序条件GAN

12.2.8.用于视频生成的双流VAN 

12.2.9.Sora：作为世界模拟器的视频生成模型



13 视频：分析/增强 

13.1.目标检测与跟踪

13.1.1 检测与跟踪

13.1.1.1 检测/分割 物体 

13.1.1.2 关联检测



13.1.2 重新识别Re-ID



13.1.3 联合检测与嵌入（JDE）







13.2.行为识别  

13.3.视频事件/异常检测

13.3.1 行为识别

13.3.2 长期循环卷积网络 Long-term Recurrent Convolutional Network 

13.3.3 C3D: 3D卷积网络

13.3.4 隐藏二流Hidden Two Stream 



13.4.视频增强

13.4.1 模糊模型Blur model 

13.4.2 模糊图像反卷积

13.4.2.1 点扩展函数Point Spread Function (PSF) 

13.4.3 深度视频去模糊Deep video deblur 

13.4.4 视频湍流效果消除Video Turbulence Effect Remove 



13.5.光流

13.5.1 什么是光流？

13.5.2 光流假设

13.5.2.1 空间相干性

13.5.2.2 时间的持久性

13.5.2.3 亮度常数约束

13.5.3 亮度常数方程

13.5.4 亮度常数约束



13.6.视频分割

13.6.1.介绍分割：像素输入，像素输出

13.6.2 Convnets进行分类

13.6.3 R-CNN

13.6.4 U-Net架构

13.6.4.1收缩阶段 

13.6.4.2扩张阶段 

13.6.4.3 U-Net总结

13.6.5 图像分割与视频分割





14 3D

14.1 3D方法 

14.1.1.飞行时间

14.1.2.激光雷达

14.1.3.结构光

14.1.4.运动构造 

14.1.4.1 介绍：从两个或多个图像重建场景几何和相机位置 

14.1.4.2 像素对应：跟踪

Harris Corner

SIFT

Scale space

Descriptor

Object Detection

SuperPoint

图像鲁棒匹配策略

(a)关键点空间离群值拒绝

(b)匹配许多特征——寻找一个好的相同图

RANSAC循环

SuperGlue

LoFTR：基于变压器的无检测器局部特征匹配

关键点跟踪：Lucas-Kanade Tracker

affine cameras



14.1.4.3投影模型 

运动中的仿射结构Affine structure

仿射模棱两可 Affine ambiguity





14.1.5.立体成像

14.1.5.1.3D视频原理

14.1.5.2.立体深度

14.1.5.3.立体匹配



14.2 3D遇上深度学习

14.2.1 介绍3D

14.2.2 三维CNN体积数据



14.3 GAN for 3D

14.3.1 从单个图像到体积

14.3.2 从单个图像到点云

14.3.3 从图像到形状

14.3.4 从图像到表面



# 8.Disclaimer

All content in this  is based solely on the contributors' personal work, Internet data.
All tips are for reference only and are not guaranteed to be 100% correct.
If you have any questions, please submit an Issue or PR.
In addition, if it infringes your copyright, please contact us to delete it, thank you.



#### Copyright © School of Electrical & Electronic Engineering, Nanyang Technological University. All rights reserved.
