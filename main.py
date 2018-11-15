# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from constants import WordType, DISPLAY_COUNT, POET_COUNT, WORD_COULD_COUNT, FAVORITE_POETS_LIST
from analyzer import plot_vectors, get_analyzer
from preprocessor import StemResult
from sklearn.cluster import KMeans


def show_counter(counter):
    for k, v in counter:
        print(k, v)

    print()


def show_wordcloud(word_dict, title=None):
    wordcloud = WordCloud(
        font_path='fonts/msyh.ttf',
        background_color='white',
        scale=2,
    ).fit_words(word_dict)
    plt.imshow(wordcloud)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()


def draw_area(figure, X, target, specific_poets, title, subplot):
    """绘制多个诗人区域图"""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    ax = plt.subplot(120 + subplot)
    x_axis = []
    y_axis = []

    for i in range(X.shape[0]):
        x_axis.append(X[i, 0])
        y_axis.append(X[i, 1])

    ax.scatter(x_axis, y_axis, c='b', marker='o')

    first_point = []
    last_point = []
    for i in range(X.shape[0]):
        if target[i] in specific_poets:
            if len(last_point):
                ax.plot([last_point[0], X[i, 0]], [last_point[1], X[i, 1]], color='r')
            else:
                first_point = [X[i, 0], X[i, 1]]
            last_point = [X[i, 0], X[i, 1]]
            ax.text(X[i, 0], X[i, 1], target[i])
    ax.plot([last_point[0], first_point[0]], [last_point[1], first_point[1]], color='r')
    plt.title(title)


def draw_compared_plot(specified_poets, group_name):
    fig = plt.figure()
    draw_area(fig, tf_idf_vector_list, poet_list, specified_poets, title='TF-IDF', subplot=1)
    draw_area(fig, w2v_vector_list, poet_list, specified_poets, title='Word2Vector', subplot=2)
    plt.suptitle('{}的距离可视化'.format(group_name))
    plt.show()

# 确定保存位置
saved_dir = os.path.join(os.curdir, "out")

# 对宋词进行分词
result = StemResult()
result = result.stem_poem("ci.song.all.json", saved_dir)

# 用word2vector和tfidf算法对所有宋词数据进行学习和分析，得到词向量
analyzer = get_analyzer(result, saved_dir)

# 装填词人的词向量信息
tf_idf_vector_list = []
tf_idf_vector_dimension_list = []
w2v_vector_list = []
w2v_vector_dimension_list = []
poet_list = []
for c in result.poet_counter.most_common(POET_COUNT):
    poet = c[0]
    index = analyzer.poets.index(poet)
    w2v_vector_list.append(analyzer.w2v_word_vector_tsne[index])
    w2v_vector_dimension_list.append(analyzer.w2v_word_vector[index])
    tf_idf_vector_list.append(analyzer.tfidf_word_vector_tsne[index])
    tf_idf_vector_dimension_list.append(analyzer.tfidf_word_vector[index])
    poet_list.append(poet)

# 绘制散点图
plot_vectors(tf_idf_vector_list, poet_list, 'tf_idf')
plot_vectors(w2v_vector_list, poet_list, 'w2v')

draw_compared_plot(["苏轼", "辛弃疾", "贺铸", "陈亮", "张元干", "刘过"], "豪放派")
draw_compared_plot(["柳永", "张先", "晏殊", "欧阳修", "秦观", "李清照"], "婉约派")
draw_compared_plot(["黄庭坚", "秦观", "晁补之", "张耒"], "苏门四学士")
draw_compared_plot(["周邦彦", "姜夔", "吴文英", "张炎", "王沂孙"], "格律派诗人")
draw_compared_plot(["晏殊", "晏几道", "欧阳修", "黄庭坚"], "江西词派")

print('统计分析')
print('-----------------')
print('统计了%s位词人的%s首词' % (len(result.poet_counter.keys()), sum(result.poet_counter.values())))
print("写作数量排名：")
most_productive_poets = result.poet_counter.most_common(DISPLAY_COUNT)
show_counter(most_productive_poets)

print("最常用的非单字词：")
cnt = 0
most_frequent_words = []
for word, count in result.word_counter.most_common():
    if cnt >= WORD_COULD_COUNT:
        break
    if len(word) > 1:
        most_frequent_words.append((word, count))
        cnt += 1
show_counter(most_frequent_words[:DISPLAY_COUNT])

most_frequent_word_dict = {}
for word in most_frequent_words:
    most_frequent_word_dict[word[0]] = word[1]

show_wordcloud(most_frequent_word_dict, '全宋词词云')

print("最常用的名词：")
most_common_nouns = result.word_property_counter_dict[WordType.NOUN].most_common(DISPLAY_COUNT)
show_counter(most_common_nouns)

print("最常用的地名：")
show_counter(result.word_property_counter_dict[WordType.PLACE].most_common(DISPLAY_COUNT))

print("最常用的形容词：")
show_counter(result.word_property_counter_dict[WordType.ADJ].most_common(DISPLAY_COUNT))

print("最常用的连词：")
show_counter(result.word_property_counter_dict[WordType.CONJ].most_common(DISPLAY_COUNT))

print("最常用的数词：")
show_counter(result.word_property_counter_dict[WordType.NUM].most_common(DISPLAY_COUNT))

print("最常用的介词：")
show_counter(result.word_property_counter_dict[WordType.PREP].most_common(DISPLAY_COUNT))

print("最常用的动词：")
show_counter(result.word_property_counter_dict[WordType.VERB].most_common(DISPLAY_COUNT))

print("**基于词向量的分析")
for word in list(most_common_nouns):
    print("与 %s 相关的词：" % word[0])
    show_counter(analyzer.find_similar_word(word[0]))

# 个人喜爱的一些词人的数据分析
for poet_name in FAVORITE_POETS_LIST:
    print("与 %s 用词相近的诗人：" % poet_name)
    print("tf-idf标准： %s" % analyzer.find_similar_poet(poet_name))
    print("word2vector标准： %s\n" % analyzer.find_similar_poet(poet_name, use_w2v=True))
    print("%s 最常用的词牌统计:" % poet_name)
    show_counter(result.poet_cipai_counter[poet_name].most_common(DISPLAY_COUNT))

    cnt = 0
    most_frequent_words = []
    for word, count in result.poet_word_counter[poet_name].most_common():
        if cnt >= WORD_COULD_COUNT:
            break
        if len(word) > 1:
            most_frequent_words.append((word, count))
            cnt += 1
    most_frequent_word_dict = {}
    for word in most_frequent_words:
        most_frequent_word_dict[word[0]] = word[1]

    # show_wordcloud(most_frequent_word_dict, poet_name)

# 聚类分析
cluster = KMeans(n_clusters=2, random_state=0)
labels = cluster.fit_predict(tf_idf_vector_dimension_list)

for i in range(2):
    print('--Group {}----'.format(i + 1))
    for j in range(len(labels)):
        if i == labels[j]:
            print(poet_list[j])

