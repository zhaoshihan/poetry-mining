import os
import pickle
<<<<<<< HEAD
import re
import json


from collections import Counter, OrderedDict
from jieba import posseg as pseg
from constants import TITLE_CONTENT_THRESHOLD, END_POET_LIST_SEPARATOR, STEM_RESULT_FILENAME

zhPattern = re.compile(u'[\u4e00-\u9fff]+')

class StemResult(object):
    """
    分词结果
    char_counter：字频统计
    poet_counter：作者计数
    word_set：词汇表
    word_counter：词汇计数
    cipai_counter: 词牌计数
    word_property_counter_dict：词汇词性
    poet_poetry_dict：解析后的结果，作者与他对应的诗
    """

    def __init__(self):
        self.word_set = set()
        self.word_counter = Counter()
        self.word_property_counter_dict = {}
        self.poet_word_counter = {}
        self.poet_cipai_counter = {}
        self.char_counter = Counter()
        self.poet_counter = Counter()
        self.poet_poetry_dict = OrderedDict()

    def add_stem_poetry(self, poet, divided_lines):
        """为poet_poetry_dict添加对象"""
        ctp = self.poet_poetry_dict.get(poet)
        if ctp is None:
            self.poet_poetry_dict[poet] = ""
        else:
            self.poet_poetry_dict[poet] += ' '
        self.poet_poetry_dict[poet] += ' '.join(divided_lines)

    @classmethod
    def _is_chinese_character(cls, c):
        return '\u4e00' <= c <= '\u9fff'

    def stem_poem(self, filename, saved_dir):
        """
        对全宋词分词
        :param: filename: 全宋词文件名
                saved_dir: 储存位置(out)
        :return:分词结果
        """
        target_file_path = os.path.join(saved_dir, STEM_RESULT_FILENAME)
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        if os.path.exists(target_file_path):
            print('load existed stem result.')
            with open(target_file_path, 'rb') as f:
                return pickle.load(f)

        else:
            print('stemming poetry...')
            poetry_count = 0
            current_poet = None
            divided_lines = []
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for poetry_item in data:
                poetry_count += 1
                if poetry_count % 1000 == 0:
                    print('%d poetry processed.' % poetry_count)
                try:
                    # 解析作者
                    current_poet = poetry_item["author"]

                    # 解析词牌
                    cipai_list = poetry_item["rhythmic"].split("・")
                    if self.poet_cipai_counter.get(current_poet) is None:
                        self.poet_cipai_counter[current_poet] = Counter()
                    # 统计词牌数
                    for cipai in cipai_list:
                        self.poet_cipai_counter[current_poet][cipai] += 1

                    # 解析诗句
                    chars = [c for c in poetry_item["paragraphs"] if self._is_chinese_character(c)]
                    for char in chars:
                        self.char_counter[char] += 1

                    cut_line = pseg.cut("".join(poetry_item["paragraphs"]))
                    for word, property in cut_line:
                        # 非中文则跳过
                        if not self._is_chinese_character(word):
                            continue
                        # 统计不同词性词频
                        if self.word_property_counter_dict.get(property) is None:
                            self.word_property_counter_dict[property] = Counter()
                        self.word_property_counter_dict[property][word] += 1
                        self.word_set.add(word)
                        self.word_counter[word] += 1

                        # 统计每个词人的用词词频
                        if self.poet_word_counter.get(current_poet) is None:
                            self.poet_word_counter[current_poet] = Counter()
                        self.poet_word_counter[current_poet][word] += 1

                        divided_lines.append(word)

                    divided_lines.append("\n")
                    # 将当前分词后的结果加入结果表中
                    self.add_stem_poetry(current_poet, divided_lines)
                    self.poet_counter[current_poet] += 1
                    divided_lines = []

                except Exception as e:
                    print('{poetry_num}-解析全宋词文件异常 {poetry_item}'.format(
                        poetry_num=poetry_count,
                        poetry_item=poetry_item,
                    ))
                    raise e
            with open(target_file_path, 'wb') as f:
                pickle.dump(self, f)
            f.close()
            print('closed')

        return self
=======

from collections import Counter, OrderedDict
from jieba import posseg as pseg


class CutResult(object):
    """
    分词结果
    char_counter：字频统计
    author_counter：作者计数
    word_set：词汇表
    word_counter：词汇计数
    word_property_counter_dict：词汇词性
    author_poetry_dict：解析后的结果，作者与他对应的诗
    """

    def __init__(self):
        self.char_counter = Counter()
        self.author_counter = Counter()
        self.word_set = set()
        self.word_counter = Counter()
        self.word_property_counter_dict = {}
        self.author_poetry_dict = OrderedDict()

    def add_cut_poetry(self, author, divided_lines):
        """为author_poetry_dict添加对象"""
        ctp = self.author_poetry_dict.get(author)
        if ctp is None:
            self.author_poetry_dict[author] = ""
        else:
            self.author_poetry_dict[author] += ' '
        self.author_poetry_dict[author] += ' '.join(divided_lines)


def _is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'


def cut_poetry(filename, saved_dir):
    """
    对全唐诗分词
    :param: filename: 全唐诗输入文件位置
            saved_location: 结果存储位置
    :return:分词结果
    """
    target_file_path = os.path.join(saved_dir, 'cut_result.pkl')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    if os.path.exists(target_file_path):
        print('load existed cut result.')
        with open(target_file_path, 'rb') as f:
            result = pickle.load(f)
    else:
        print('begin cutting poetry...')
        result = CutResult()
        line_count = 0
        current_author = None
        divided_lines = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
                if line_count % 5000 == 0:
                    print('%d lines processed.' % line_count)
                # if line_count > 10000:
                #     break
                try:
                    if line.strip() == "":
                        continue
                    # 解析作者
                    if "【" in line:
                        header = line.split()[1]
                        author = header[header.find("】") + 1:].strip()
                        result.author_counter[author] += 1
                        divided_lines.append("\n")
                        # 将当前分词后的结果加入结果表中
                        if current_author is not None:
                            result.add_cut_poetry(current_author, divided_lines)
                            divided_lines = []
                        current_author = author
                        continue
                    # 解析诗句
                    chars = [c for c in line if _is_chinese(c)]
                    for char in chars:
                        result.char_counter[char] += 1
                    cut_line = pseg.cut(line)
                    for word, property in cut_line:
                        if not _is_chinese(word):
                            continue
                        if result.word_property_counter_dict.get(property) is None:
                            result.word_property_counter_dict[property] = Counter()
                        result.word_property_counter_dict[property][word] += 1
                        result.word_set.add(word)
                        result.word_counter[word] += 1
                        divided_lines.append(word)
                except Exception as e:
                    print("%d-解析全唐诗文件异常 %s" % (line_count, line))
                    raise e
        # 加入最后一次解析的结果
        result.add_cut_poetry(current_author, divided_lines)
        with open(target_file_path, 'wb') as f:
            pickle.dump(result, f)
    return result
>>>>>>> 443227c8fa5efae27ecbb93237541a89ef3bb7e3
