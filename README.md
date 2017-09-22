# deepleraning.ai-course
刚开始学习Andrew Ng在Coursera的deeplearning课程，在这里将一些编程作业做一些记录。
如果有任何错误请批评指正，欢迎联系**zpzhang1995@gmail.com**

## 课程目录
## Neural Networks and Deep Learning
   **Week1** Introduction to deep learning   
   **Week2** Neural Networks Basics   
   **Week3** Shallow Neural networks    
   **Week4** Deep Neural Networks 

## Improving Deep Neural Networks  
   **Week1** Practical aspects of Deep Learning(Initialization-Regularization-Gradient Checking)  
   
   **Week2** Optimization algorithms  
   
    
           
        
      
   努力学习中...

## 总结
#### 1. DNN不一定比传统方法好，效果也与数据量有关，数据量少的时候可能传统方法表现更好（当然也可能是神经网络）
####  ![image](https://github.com/JudasDie/deeplearning.ai-course/raw/master/images_md/DataScalevsMethod.png)   

#### 2. Cost Function采用的是交叉熵，没有用均方差，凸优化问题
####  ![image](https://github.com/JudasDie/deeplearning.ai-course/raw/master/images_md/CostFunction.png)    

#### 3. 常用的四种激活函数：Sigmoid,tanh,ReLU,leakey ReLU,可以推到一下其求导
![image](https://github.com/JudasDie/deeplearning.ai-course/raw/master/images_md/ActivationFunction.png)   

#### 4. DNN前向传播和后向传播整体流程
![image](https://github.com/JudasDie/deeplearning.ai-course/raw/master/images_md/Propagation.png)   

#### 5. Project2 week1 *Practical aspects of Deep Learning* 的一些总结
- 除非过拟合，否则Dropout不是一定要用的（可以先看看Cost Function变化趋势）
- 可以对每层设置不同Dropout Rate，参数多的设置稍微大些 
- 增加数据集的方法：水平翻转图片，图片裁剪等对原始图片做出一定变化  
- 一般要进行**输入归一化**，这样梯度下降的速度可能相对较快  
- 初始化时候不要设置过大，不然梯度会很小（程序中有相关体现，程序中用的是He-Initialization,一定程度缓解梯度消失或梯度爆炸)  
- Gradient Checking对找到bug很有参考意义 

#### 5. Project2 week2 *Optimization algorithms* 的一些总结
- 随机梯度下降不会收敛在最优位置，会在最优位置附近震动
- 如果样本数目比较小(<2000 通常)不需要使用mini-batch
- mini-batch大小通常16,64.128,256等（2的次方）（由于计算机的存储规则）
- 想理解Momentum最好去了解一下指数加权平均（课程视频讲的很详细）
- 当维度较高时，可能局部最优的困扰并不存在，问题可能是到达鞍部时候梯度很小，这也是Momentum的作用（过去听课程其他课程时候一直强调的是局部最优问题）
- Adam通常的参数取值：beta1=0.9（Momentum）,beta2=0.999(RMS prop) 
- 一定要注意Adam更新参数时分母加一个eps，原来程序结果一支不对就是这里炸了

## 说明
如果你感觉这个仓库对你有用或者起到了某些帮助，请点亮一个小星星或者fork哦，不要做伸手党啦  

虽然Andrew在网易云课堂提供了免费视频，但是还是建议像我一样刚开始学的通过Coursera学习。因为Coursera上编程练习在网易上并不能用。如果您是和我一样经济不是很宽裕的学生(Sad)，可以选择7天免费或者旁听，不影响做题，只是没有人为打分而已。


### 一起学习，一起进步！
### 感谢Andrew Ng!
