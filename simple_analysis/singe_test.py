# coding=gbk
"""
    一个个分析问题和答案
"""
import math
import os
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.parse import stanford
from collections import defaultdict
import preprocess.stanford_parser as sp
import utils.data_path as dp

def __compute_counts(stories):
    """
    统计每个passage中出现词语的个数
    :param stories: list(passage)  未分词
    :return: 
    """
    counts = defaultdict(lambda: 0.0)
    for passage in stories:
        for token in passage:
            counts[token] += 1.0
    return counts


def __compute_inverse_counts(stories):
    """
    对所有的passage来计算counts
    :param stories: list(passage)
    :return: 
    """
    counts = __compute_counts(stories)
    icounts = {}
    for token, token_count in counts.items():
        icounts[token] = np.log(1.0 + 1.0 / token_count)
    return icounts

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

def __IC(passage, word):
    if word == "broncos":
        print(word,passage.count(word))
    Cw = passage.count(word)  # passage包含word的个数
    return math.log(1 + 1 / float(Cw))

def __slide_window(passage_words,question_words,candidate_answer_list,icounts):
    # 对于每一个候选答案
    true_answer = []
    for candidate_answer in candidate_answer_list:
        candidate = candidate_answer  #果然是后面extend惹的祸！！
        print("candidate:",candidate)
        print("question_words:",question_words)
        S = set(candidate + question_words)  # question和answer word的并集
        S_len = len(S)  #相当于window size
        sw = []
        for j in range(len(passage_words)):  # 对于passage中的每个词
            temp = 0
            try:
                for w in range(S_len):  # 对于并集中的每个词
                    if passage_words[j + w] in S:
                        temp += icounts[passage_words[j + w]]
                    else:
                        temp += 0
            except IndexError:
                pass
            sw.append(temp)
        true_answer.append(max(sw))    #每个答案的slide window值

    return true_answer


def single():
    # 添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
    os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
    os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
    # 为JAVAHOME添加环境变量
    java_path = dp.JAVA_PATH
    os.environ['JAVAHOME'] = java_path
    PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)

    #文章
    passage = word_tokenize("""Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the "golden anniversary" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as "Super Bowl L"), so that the logo could prominently feature the Arabic numerals 50.""".lower())
    #passage = word_tokenize("""The Panthers finished the regular season with a 15–1 record, and quarterback Cam Newton was named the NFL Most Valuable Player (MVP). They defeated the Arizona Cardinals 49–15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995. The Broncos finished the regular season with a 12–4 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20–18 in the AFC Championship Game. They joined the Patriots, Dallas Cowboys, and Pittsburgh Steelers as one of four teams that have made eight appearances in the Super Bowl.""")


    #分词
    sentence_one = word_tokenize("the american football conference afc champion denver broncos defeated the national football conference nfc champion carolina panthers 24-0 to earn their third super bowl title.".lower())
    sentence_two = word_tokenize("as this was the 50th super bowl, the league emphasized the golden anniversary with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each super bowl game with roman numerals under which the game would have been known as super bowl, so that the logo could prominently feature the arabic numerals 50.".lower())
    question = word_tokenize("which nfl team represented the afc at super bowl 50?".lower())
    # question = word_tokenize("Who did the Panthers beat in the NFC Championship Game?")
    # sentence_one = word_tokenize("They defeated the Arizona Cardinals 49–15 in the NFC Championship Game and advanced to their second Super Bowl appearance since the franchise was founded in 1995.")
    # sentence_two = word_tokenize("The Broncos finished the regular season with a 12–4 record, and denied the New England Patriots a chance to defend their title from Super Bowl XLIX by defeating them 20–18 in the AFC Championship Game.")

    #去停用词
    sentence_one = [w for w in sentence_one if (w not in stopwords.words('english'))]
    sentence_two = [w for w in sentence_two if (w not in stopwords.words('english'))]
    question = [w for w in question if (w not in stopwords.words('english'))]

    #正确答案
    answer = "denver" #或者broncos

    #生成候选答案
    sentence_one_candidate = sp.getConstituents(PAESER, " ".join(sentence_one))
    sentence_two_candidate = sp.getConstituents(PAESER, " ".join(sentence_two))

    print("sentence_one_candidate:")
    for i in sentence_one_candidate:
        a = ""
        for j in i:
            a = a+" "+str(j)
        print(a)
    print("sentence_one_candidate:",sentence_one_candidate)
    print("sentence_two_candidate:",sentence_two_candidate)
    print("passage:",passage)

    """
    unigram可能更接近于词语的匹配，bigram则更接近于词语的搭配（bigram更考虑语法？）
    下面的实验中unigram overlap为5，bigram overlap为1---只进行word_tokenize
    下面的实验中unigram overlap为4(sentence_one)，3(sentence_two)，bigram overlap为1---进行word_tokenize和去停用词
    """
    sentence_one_ngram = __getNgrams(sentence_one,2)
    sentence_two_ngram = __getNgrams(sentence_two, 2)
    question = __getNgrams(question,2)
    overlap_one = __overlap(sentence_one_ngram,question)
    overlap_two = __overlap(sentence_two_ngram,question)

    print(overlap_one,overlap_two)

    #计算inverse counts
    icounts = __compute_inverse_counts(list([passage]))
    # 使用slide windows
    true_answer = __slide_window(passage, question, sentence_one_candidate,icounts)

    pd_candidate = pd.DataFrame(np.array([sentence_one_candidate,true_answer]).T,columns=['candidate','slide_window'])
    print(pd_candidate)

    pd_candidate = pd_candidate[pd_candidate['slide_window'] == pd_candidate['slide_window'].max(0)]  #获取最大的maximal overlap
    print(pd_candidate['candidate'])


if __name__ == '__main__':
    single()