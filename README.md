# 基于Gender Classification Dataset 数据集的西方人种的性别分类 超轻量级 CNN模型
https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset
### 结果
分类测试集准确率可以达到 90.84899991415573%  （还可以更好，但是采用了早停策略 ，没有进一步的训练）
### 训练
训练基于4060 显卡
训练时长10min以内完成
### 架构：
_gcNet卷积神经网络_

架构如下：

卷积层（Convolutional Layers）：
第一卷积层：使用3个输入通道（假设是RGB图像），16个输出通道，3x3的卷积核。
第二卷积层：从16个输入通道到32个输出通道，3x3的卷积核。
第三卷积层：从32个输入通道到64个输出通道，3x3的卷积核。
每个卷积层后面都跟着一个ReLU激活函数。

池化层（Pooling Layers）：
最大池化层（Max Pooling）：每个卷积层后都紧跟一个2x2最大池化操作，用于降低特征图的空间维度（高度和宽度），没有填充（padding=0）。
全连接层（Fully Connected Layers）：
第一全连接层：从7488个输入特征到512个输出特征。
第二全连接层：从512个输入特征到128个输出特征。
第三全连接层（输出层）：从128个输入特征到2个输出特征。
每个全连接层（除了输出层）后都有一个ReLU激活函数和一个Dropout层。

Dropout层（Dropout Layers）：
在全连接层之间使用，以给定的dropout_rate（默认为0.5）随机丢弃一部分神经元，以防止过拟合。
输出激活函数：
最后的输出通过一个Sigmoid函数，这常用于二分类问题，使得模型输出可以解释为属于某类的概率。
输入和输出：
模型接受形状为(批次大小, 3, 90, 120)的张量作为输入，其中3表示图像的颜色通道数，90和120分别是图像的高度和宽度。
输出是形状为(批次大小, 2)的张量，表示两个类别的预测概率。


### 运行方法
1. 下载项目到本地
   `git clone [git链接]`
2. 切换到项目的根目录,安装依赖
	`pip install -r requirements.txt`
	python 3.9
  CDUA 11.8
	torch安装失败请尝试：  `conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia`
	请注意numpy版本：要求小于1.20版本 否则将会报错！
3.运行exam_cuda_exist.py 查看电脑的cudnn和cuda的版本，以在程序环境出错的时候查询电脑的cuda和cudnn和torch的版本的匹配性，如果确信环境正常，此步可以跳过
4.下载数据集到本地根目录下的dataset目录中，解压得到archive 文件夹
5.运行datafiles_creating目录下的raw_filelist.py 创建数据集的索引，对数据集留分 （该代码中创建数据集train/val 比率为0.9 默认不用修改）
6.运行gender_main.py代码，他是训练和测试的代码的集合：
[1]如果想要进行训练，请注释resume = "best_checkpoint.pth.tar" ，解除 # resume = None 的注释；
[2]在训练结束之后，可以解除注释resume = "best_checkpoint.pth.tar"，添加 resume = None 的注释；
[3]默认设置当验证集准确率超过90% 自动停止训练，启动测试模式，开始进行测试集的评估；如果想要更好的准确率，请调节`if acc>90:`代码；
[4]如果想要单独的进行模型的测试，请设置    evaluate = True   这样可以跳过训练，但是同时需要设置resume 的路径，为你要测试的模型的路径；


### 训练过程图



### 数据集结构：
train:42340
val:4669
test:11649
all:58658 


