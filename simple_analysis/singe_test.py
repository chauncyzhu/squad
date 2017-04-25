# coding=gbk
"""
    一个个分析问题和答案
"""
import os
import utils.data_path as dp
from nltk.parse import stanford
from nltk import word_tokenize
import candidate_answer.stanford_parser as sp


def __getNgrams(input, n):
    """
    获取ngram模型
    :param input: list，已经分好词并去除了stopwords
    :param n: ngram中的n
    :return: 返回ngram模型
    """
    output = [] # 构造字典
    for i in range(len(input)-n+1):
        ngramTemp = " ".join(input[i:i+n])  #.encode('utf-8')
        output.append(ngramTemp)
    return output

def __overlap(dict_one,dict_two):
    """
    计算两个dict之间的overlap
    :param dict_one:词典1
    :param dict_two:词典2
    :return:返回两个之间overlap的值
    """
    return len(set(dict_one).intersection(dict_two))

def single_test():
    # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
    os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
    os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
    # 为JAVAHOME添加环境变量
    java_path = dp.JAVA_PATH
    os.environ['JAVAHOME'] = java_path
    PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)

    single_line = "the american football conference (afc) champion denver broncos defeated the national football conference (nfc) champion carolina panthers 24??0 to earn their third super bowl title."

    constituents_list = sp.getConstituents(PAESER, single_line)  # 注意，这里实际上是三层嵌套列表，需要两个循环进行解析
    print(constituents_list)
    sentence_one = word_tokenize("the american football conference (afc) champion denver broncos defeated the national football conference (nfc) champion carolina panthers 24??0 to earn their third super bowl title.")
    sentence_two = word_tokenize("as this was the 50th super bowl, the league emphasized the ""golden anniversary"" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each super bowl game with roman numerals (under which the game would have been known as ""super bowl l""), so that the logo could prominently feature the arabic numerals 50.")

    question = word_tokenize("which nfl team represented the afc at super bowl 50?")

    answer = "denver" #或者broncos

    sentence_one_ngram = __getNgrams(sentence_one,1)
    sentence_two_ngram = __getNgrams(sentence_two, 1)
    overlap = __overlap(sentence_one_ngram,sentence_two_ngram)

    print(overlap)

if __name__ == '__main__':
    single_test()