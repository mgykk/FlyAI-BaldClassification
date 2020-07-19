# FlyAI-BaldClassification

一个图像识别分类比赛

## 数据处理

- 图片输入大小

依据短边裁剪将输入尺寸限制在176*176.

- 数据增强

采用了基于albumentations库的数据增强，主要有以下几种:

1. 重新调整图像大小;
2. 水平旋转;
3. 随机应用仿射变换;
4. 图像的弹性变形;
5. 添加高斯噪声增加鲁棒性.

可选数据增强方法：

1. Mix_up:同一batch做线性混合;
2. CutMix:从一幅图中随机裁剪处一个ROI覆盖到当前区域;
3. FMix:从随机图像中剪切出任意形状的部分，并将其粘贴到相关图像上，不同于一般的剪切和粘贴，其需要掩膜来定义图像哪些部分需要考虑，通过对傅里叶空间采样的低频图像进行阈值处理得到掩膜.

经过测试发现三种方法并不适合这个识别任务，会导致精度丢失。

- 数据集划分

采用7折交叉验证的方法，对数据集划分为7份分别作为测试集进行交叉验证.

## 模型选择与设计

在网络尾部添加全局池化层输出特征向量，向网络分类头结构中插入两层全连接层，并嵌入Dropout层，防止网络过拟合，训练时，使用平台提供的预训练模型，这可以使得网络模型更快地收敛.

考虑到数据集大小，测试了efficientnet_b2，mobilenetV2，inceptionV4，resnet50，senet50以及seresnext50的性能.

 根据模型指标对seresnext50和efficientnet-b2分别进行7折交叉验证，总共得到14个模型，利用均值法进行模型集成.

## 超参数设置

- 借鉴detectron2训练采用的学习率预热(warm up)方法，进行简单实现，进行5轮预热，使得模型健康收敛.
- 采用标签平滑(label smoothing)，smoothing=0.2.
- 优化器选择：RAdam，lr=0.003，Lookahead方法.
- 采用余弦退火方法.

学习率图像：

<img src="https://github.com/mgykk/FlyAI-BaldClassification/blob/master/images/Figure_1.jpg" style="zoom:80%;" />

## TTA

Test time augmentation(TTA)，测试数据增强，是在测试阶段时，将输入的测试数据进行，翻转、旋转操作等数据增强，并最后对同一样本的不同数据增强的结果根据任务需求进行例如平均，求和等数据处理。
图像分类比赛常用涨点方法。

## 模型得分

![](https://github.com/mgykk/FlyAI-BaldClassification/blob/master/images/result.png)
