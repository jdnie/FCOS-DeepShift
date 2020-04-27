# 说明

DeepShift把卷积中的乘法转换成shift+sgn，本质上是n位二进制小数的round。

原始值越小精度越高，原始值越大精度越低，例如round(0.6) = 0.5，error=0.1，round(3) = 2，error=1。

如果训练的模型的weight都是较小的值，这种方式能实现保证模型原始精度的量化加速。

官方的代码只有分类的，拿FCOS测试了一下在检测中的效果。

测试发现最好前期先训练一个正常的检测模型，然后再转换成DeepShift模型并finetune，finetune时量化范围可以逐渐缩小，官方默认的量化范围是(-15, 0)，正常的检测模型weight如果大于1比较多，误差会较大，个人推荐(-10, 5)，对应浮点范围大概是(2^-10 ~2^5)，大概占用16bit。

## 安装

参考FCOS和DeepShift。

https://github.com/mostafaelhoushi/DeepShift

https://github.com/tianzhi0549/FCOS

