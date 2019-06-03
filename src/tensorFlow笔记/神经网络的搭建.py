# TensorFlow做加法
import tensorflow as tf #引入模块
#定义一个张量等于[1.0,2.0]
a = tf.constant([1.0, 2.0]) 
#定义一个张量等于[3.0,4.0]
b = tf.constant([3.0, 4.0]) 
#实现 a 加 b 的加法
result = a+b 
#打印出结果
# Tensor(“add:0”, shape=(2, ), dtype=float32)
# 意思为 result 是一个名称为 add:0 的张量，shape=(2,)表示一维数组长度为 2，
# dtype=float32 表示数据类型为浮点型 
print(result)

# x1、 x2 表示输入， w1、 w2 分别是 x1 到 y 和 x2 到 y 的权重， y=x1*w1+x2*w2。
# 我们实现上述计算图
#定义一个 2 阶张量等于[[1.0,2.0]]
x = tf.constant([[1.0, 2.0]]) 
#定义一个 2 阶张量等于[[3.0],[4.0]]
w = tf.constant([[3.0], [4.0]]) 
#实现 xw 矩阵乘法
y = tf.matmul(x, w)
 #打印出结果 
#  Tensor(“matmul:0”, shape(1,1), dtype=float32)，
# 从这里我们可以看出， print 的结果显示 y 是一个张量，只搭建承载计算过程的
# 计算图，并没有运算，如果我们想得到运算结果就要用到“会话 Session()”了
print(y)

#定义一个 2 阶张量等于[[1.0,2.0]]
x = tf.constant([[1.0, 2.0]]) 
#定义一个 2 阶张量等于[[3.0],[4.0]]
w = tf.constant([[3.0], [4.0]]) 
 #实现 xw 矩阵乘法
y = tf.matmul(x, w)
 #打印出结果
print(y)
 #执行会话并打印出执行后的结果  
with tf.Session() as sess:
    print(sess.run(y))
# 打印出Tensor(“matmul:0”, shape(1,1), dtype=float32)
# [[11.]]
# 我们可以看到，运行 Session()会话前只打印出 y 是个张量的提示，运行 Session()
# 会话后打印出了 y 的结果 1.0*3.0 + 2.0*4.0 = 11.0

# 神经网络的参数： 是指神经元线上的权重 w，用变量表示，一般会先随机生成
# 生成参数的方法是w = tf.Variable(生成方式)，把生成的方式写在括号里。
# 神经网络中常用的生成随机数/数组的函数有：
# tf.random_normal() 生成正态分布随机数
# tf.truncated_normal() 生成去掉过大偏离点的正态分布随机数
# tf.random_uniform() 生成均匀分布随机数
# tf.zeros 表示生成全 0 数组
# tf.ones 表示生成全 1 数组
# tf.fill 表示生成全定值数组
# tf.constant 表示生成直接给定值的数组
# 1.生成正态分布随机数，形状两行三列， 标准差是 2， 均值是 0， 随机种子是 1
w=tf.Variable(tf.random_normal([2,3],stddev=2, mean=0, seed=1))
# 2.表示去掉偏离过大的正态分布， 也就是如果随机出来的数据偏离平均值超过两个
# 标准差，这个数据将重新生成
w=tf.Variable(tf.Truncated_normal([2,3],stddev=2, mean=0, seed=1))
# 3.从一个均匀分布[minval maxval)中随机采样，注意定义域是左闭右开，即包含 minval，不包含 maxval
w=tf.random.uniform(shape=7,minval=0,maxval=1,dtype=tf.int32, seed=1)
# 4.除了生成随机数， 还可以生成常量。
tf.zeros([3,2],int32)
tf.ones([3,2],int32)
tf.fill([3,2],6)
tf.constant([3,2,1])

# 神经网络的实现过程：
# 1、准备数据集，提取特征，作为输入喂给神经网络（ Neural Network， NN）
# 2、搭建 NN 结构，从输入到输出（先搭建计算图，再用会话执行）
# （NN 前向传播算法===> 计算输出）
# 3、大量特征数据喂给 NN，迭代优化 NN 参数
# （NN 反向传播算法 ===>优化参数训练模型）
# 4、使用训练好的模型预测和分类
# √前向传播过程的 tensorflow 描述：
# √变量初始化、计算图节点运算都要用会话（ with 结构）实现
# with tf.Session() as sess:
# sess.run()
# √变量初始化：在 sess.run 函数中用 tf.global_variables_initializer()汇
# 总所有待优化变量。
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
# √计算图节点运算：在 sess.run 函数中写入待运算的节点
# sess.run(y)
# √用 tf.placeholder 占位，在 sess.run 函数中用 feed_dict 喂数据
# 喂一组数据：
# x = tf.placeholder(tf.float32, shape=(1, 2))
# sess.run(y, feed_dict={x: [[0.5,0.6]]})
# 喂多组数据：
# x = tf.placeholder(tf.float32, shape=(None, 2))
# sess.run(y, feed_dict={x: [[0.1,0.2],[0.2,0.3],[0.3,0.4],[0.4,0.5]]})

# ①用 placeholder 实现输入定义（ sess.run 中喂入一组数据）的情况
# 第一组喂体积 0.7、 重量 0.5
#coding:utf-8
import tensorflow as tf
#定义输入和参数
x=tf.placeholder(tf.float32,shape=(1,2))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
#用会话计算结果
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
print("y in tf3_3.py is:\n",sess.run(y,feed_dict={x:[[0.7,0.5]]}))

# ②用 placeholder 实现输入定义（ sess.run 中喂入多组数据）的情况
# 第一组喂体积 0.7、重量 0.5，第二组喂体积 0.2、重量 0.3，第三组喂体积 0.3 、
# 重量 0.4，第四组喂体积 0.4、重量 0.5.
#coding:utf-8
import tensorflow as tf
#定义输入和参数
x=tf.placeholder(tf.float32,shape=(None,2))
w1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
#定义前向传播过程
a=tf.matmul(x,w1)
y=tf.matmul(a,w2)
#用会话计算结果
with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
print("y in tf3_4.py is:\n",sess.run(y,feed_dict={x:[[0.7,0.5],
[0.2,0.3],[0.3,0.4], [0.4,0.5]]}))

# 反向传播训练方法： 以减小 loss 值为优化目标，有梯度下降、 momentum 优化
# 器、 adam 优化器等优化方法。
# 这三种优化方法用 tensorflow 的函数可以表示为：
# train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# train_step=tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
# train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss)
# 三种优化方法区别如下：
# ①tf.train.GradientDescentOptimizer()使用随机梯度下降算法，使参数沿着
# 梯度的反方向，即总损失减小的方向移动，实现更新参数
# ②tf.train.MomentumOptimizer()在更新参数时，利用了超参数，参数更新公式
# 𝑑𝑖 = 𝛽𝑑𝑖−1 + 𝑔(𝜃𝑖−1)
# 𝜃𝑖 = 𝜃𝑖−1 − 𝛼𝑑𝑖
# 其中， 𝛼为学习率，超参数为𝛽， 𝜃为参数， 𝑔(𝜃𝑖−1)为损失函数的梯度。
# ③tf.train.AdamOptimizer()是利用自适应学习率的优化算法， Adam 算法和随
# 机梯度下降算法不同。随机梯度下降算法保持单一的学习率更新所有的参数，学
# 习率在训练过程中并不会改变。而 Adam 算法通过计算梯度的一阶矩估计和二
# 阶矩估计而为不同的参数设计独立的自适应性学习率

# 我们最后梳理出神经网络搭建的八股， 神经网络的搭建课分四步完成：准备工作、
# 前向传播、反向传播和循环迭代。
# √0.导入模块，生成模拟数据集；
import tensorflow as tf
import numpy as np
# 常量定义
BATCH_SIZE = 8
seed = 23455
rng = np.random.RandomState(seed)
# 生成数据集
X = rng.rand(32,2)
Y = [[int(x0+x1<1)] for (x0,x1) in X]
print("X:\n",X)
print("Y:\n",Y)
# √1.前向传播：定义输入、参数和输出
x= tf.placeholder(tf.float32,shape=(None,2))
y_= tf.placeholder(tf.float32,shape=(None,1))
w1= tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2= tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))
a= tf.matmul(x,w1)
y= tf.matmul(a,w2)
# √2. 反向传播：定义损失函数、反向传播方法
loss= tf.reduce_mean(tf.square(y-y_))
train_step= tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step= tf.train.MomentumOptimizer(0.001,0.9).minimize(loss)
# train_step= tf.train.AdamOptimizer(0.001).minimize(loss)
# √3. 生成会话，训练 STEPS 轮
with tf.session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=3000
    for i in range(STEPS):
        start = (i*BATCH_SIZE)%32
        end= start+BATCH_SIZE
        sess.run(train_step, feed_dict={x:X[start:end],y_:Y[start:end]})
        if i%500==0:
            total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
            print("After %d training steps,loss on all data is %g"%(i,total_loss))

