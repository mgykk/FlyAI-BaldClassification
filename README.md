# FlyAI-BaldClassification

一个图像识别分类比赛

## 数据处理

- 图片输入大小
- 数据增强

采用了基于albumentations库的数据增强，主要有以下几种:

1. 重新调整图像大小;
2. 水平旋转;
3. 随机应用仿射变换;
4. 图像的弹性变形;
5. 添加高斯噪声增加鲁棒性;
6. 随机裁剪区域进行0像素填充.

可选数据增强方法：

1. Mix_up:同一batch做线性混合;
2. CutMix:从一幅图中随机裁剪处一个ROI覆盖到当前区域;
3. FMix:从随机图像中剪切出任意形状的部分，并将其粘贴到相关图像上，不同于一般的剪切和粘贴，其需要掩膜来定义图像哪些部分需要考虑，通过对傅里叶空间采样的低频图像进行阈值处理得到掩膜.

- 数据集划分

采用6折交叉验证的方法，对数据集按照5：1的比例划分为训练集与测试集及进行交叉验证.

## 模型选择与设计

在网络分类头结构中插入两层全连接层，并嵌入Dropout层，防止网络过拟合，训练时，使用基于imagenet的预训练模型，这可以使得网络模型更快地收敛.

考虑到数据集大小，测试了efficientnet_b2，mobilenetV2，inceptionV4，resnet50，senet50以及seresnext50的性能.

 根据模型指标对seresnext50和efficientnet-b2分别进行6折交叉验证，总共得到12个模型，利用均值法进行模型集成.

## 超参数设置

- 借鉴detectron2训练采用的学习率预热(warm up)方法，进行简单实现，进行5轮预热，使得模型健康收敛.
- 采用标签平滑(label smoothing)，smoothing=0.2.
- 优化器选择：RAdam，lr=0.003，Lookahead方法.
- 采用余弦退火方法.

## TTA

Test time augmentation(TTA)是对测试数据集进行数据扩展的测试时增强方法。

考虑到秃头小宝贝的特殊性，垂直翻转对其没有任何现实意义，在本次本赛只通过水平翻转来增加测试集，对其取平均，得到最终得分。

## 模型得分

![image-20200710165514083](https://github.com/mgykk/FlyAI-BaldClassification/blob/master/images/result.png)
