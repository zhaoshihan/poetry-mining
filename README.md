# 宋词数据分析

特别感谢[router8008](https://github.com/router8008/poetry-mining)对本项目的启发，以及[jackeyGao](https://github.com/chinese-poetry/chinese-poetry)等用户对宋词的爬取收集。

这个项目尝试使用一些文本特征分析的方法，对21000+首宋词进行数据分析。

本项目是为了完成商务智能课程的最后作业，说明文档详见introduction.md，以下文档内容尊重原作者的格式不变。

秉承fork源代码的开源精神，这里一切代码供学习之用，全部公开

## 样例

具体可见result.ipynb文件中的展示。

## 运行样例

#### 依赖环境

程序使用了Anaconda的集成python环境进行开发，用到了内部例如`numpy`, `sklearn`, `matplotlib`等机器学习和数学处理库。

另外，还需要如下环境：

```shell
pip3 install jieba 		#用于中文分词
pip3 install gensim		#用于计算word2vec
pip3 install wordcloud	#用于生成词云
```

#### 运行方法

当配置好环境后，这样运行样例：

```shell
python3 main.py
```

## 实现介绍

#### 词向量

首先，数据分析的一个重要部分是计算每个诗人所写的诗的集合词向量，可以看作是诗人的“文风”。关于词向量的计算，使用了两种方法：

- **tf-idf**

  通过文本中词语的的tf-idf值计算诗人的用词特征，之后计算他们的余弦相似度，推荐一篇可以参考相关知识的blog：[sklearn文本特征提取](http://blog.csdn.net/xiaoxiangzi222/article/details/53490227)。

  这种方法的的弊端在于没有考虑到词语之间的关联性，比如说"青"，"白"之间的关联程度，和"青"，"衣"之间的关联度肯定是不同的，所以引入了下一种计算词向量的方法。


- **word2vec**

  Word2Vec的基本思想是把自然语言中的每一个词，表示成一个统一意义统一维度的短向量。通过word2vec训练后，得到每个词语的词向量，再通过求和平均的方法获得文本的词向量，则可以得到每个诗人的词向量。当然，求平均肯定不是一个计算文档词向量的最优方法，此处有待改进。

  关于这个方法可以参考：[word2vec&doc2vec词向量模型](http://www.cnblogs.com/maybe2030/p/5427148.html)


#### 数据降维

为了便于显示，将计算结果降维，用到了t-SNE算法。

关于t-SNE算法可以参考：[t-SNE聚类算法实践指南](https://yq.aliyun.com/articles/70733)，其中的代码实现可以参考sklearn的官方文档：[Manifold learning on handwritten digits: Locally Linear Embedding, Isomap…](http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html#sphx-glr-auto-examples-manifold-plot-lle-digits-py)


---


### 欢迎对该样例进行补充和修改！

