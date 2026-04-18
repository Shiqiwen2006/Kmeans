# K-means

### 第二份作业-Kmeans检测异常值

文件说明：
- main.ipynb 题干和运行代码
- main.py 运行代码（从ipynb后几个cell里面截出来的）
- train.py 训练脚本（这个作业我在线上训练的）
- data里面两份文件 训练数据
- results里面存了模型

###评分标准

最终得分遵循以下映射表：

$$
\begin{array}{c|c}
\text{异常点数量 } n & \text{最终得分} \\
\hline
n \leq 10 & 50 + 2 \times n \\
10 < n \leq 20 & 70 + n - 10 \\
20 < n \leq 30 & 80 + (n - 20) \times 1.5 \\
n > 30 & 80 + (n - 20) \times 1.5 - miss
\end{array}
$$

其中 $miss$ 为误判断的异常值点。

现在的平台测试只告诉我模型评分过低。。。
