# coding=gbk
"""
    滑动窗口分析--机器阅读，由于candidate answer生成方式还未弄懂，因此先写分析的代码
"""
import math
import os
import time

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.parse import stanford

import utils.data_path as dp
from candidate_answer import stanford_parser as sp
from preprocess.squadReader import SquadReader

#添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
#为JAVAHOME添加环境变量
java_path = dp.JAVA_PATH
os.environ['JAVAHOME'] = java_path
PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)
N_GRAM = 2

def candidate_answer(sentences):
    """
    根据passage生成候选答案，返回候选答案以及所在的句子
    :param sentences: 文章中的一个个句子
    :return: 传引 答案以及所在的句子，pandas dataframe--candidate answer对应的sentence不应该包括candidate answer
    """
    candidate_dict = {}
    print("--------begin find all candidate answers.---------")
    for sen in sentences:  #这个是一个list
        if sen:
            print("****** sentence in the passage ******")
            constituents = sp.getConstituents(PAESER,sen)  #实际上解析的时候也会花费比较多的时间
            if sen not in candidate_dict:
                candidate_dict[sen] = constituents

    pd_candidate = pd.DataFrame(columns=['candidate_answer','sentence'])
    count = 0
    for key,value in candidate_dict.items():
        for va in value:
            pd_candidate.loc[count] = [va,key.replace(" ".join(va),'')]  #去掉candidate answer本身
            count += 1
    print("****** all candidate answer ******")
    print("--------end find all candidate answers.---------")
    return pd_candidate

def clean_data(data):
    """
    清理数据，分词、去除停用词
    :param data: 数据 string
    :return: data list
    """
    data = word_tokenize(data)
    data = [w for w in data if (w not in stopwords.words('english'))]
    return data

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

def __maximal_overlap(pd_candidate,question):
    """
    获得每个候选answer所在的最大的overlap的sentence
    :param pd_candidate: dataframe 候选答案，包括了candidate_answer以及对应的sentence
    :param question: question只有一个
    :return: pd_candidate 因为最后进行了筛选，因此需要return 返回所有最大overlap的candidate answer(bool表示)
    """
    #print("question:",question)
    def f(x):
        # 计算sentence和question的unigram/bigram(应该是或者的意思，这里计算bigram) overlap
        question_bigram = __getNgrams(question, N_GRAM)
        sentence_bigram = __getNgrams(x, N_GRAM)
        overlap = __overlap(question_bigram, sentence_bigram)  # 两个句子之间的overlap
        return overlap

    pd_candidate['overlap'] = pd_candidate['sentence'].apply(f)
    pd_candidate = pd_candidate[pd_candidate['overlap'] == pd_candidate['overlap'].max(0)]  #获取最大的maximal overlap

    print("****** maximal overlap candidate answer ******")
    #print(pd_candidate)
    return pd_candidate

def __IC(passage,word):
    Cw = passage.count(word)  #passage包含word的个数
    return math.log(1+1/float(Cw))


def slide_window(row):
    """
    通过slide window选出最合适的answer
    :param row: pd_data的一行，包含了passage、passage_words、sentences、question、question_words
    :return:final_answer 返回最终的答案 pd_candidate pandas dataframe
    """
    passage = row['passage']  #都是唯一
    passage_words = row['passage_words']
    question = row['question']  #都是唯一
    question_words = row['question_words']

    #找出所有候选答案
    pd_candidate = candidate_answer(row['sentences'])

    #根据unigram/bigram找出所有sentence和question最大的overlap对应的candidate answer
    pd_candidate = __maximal_overlap(pd_candidate,question)  #找到每个answer最大overlap所在的sentence，此时pd_candidate已经经过筛选

    pd_candidate['slide_window'] = list(range(len(pd_candidate)))

    for answer in row["answer_words"]:
        print("pd_candidate candidate_answer:",pd_candidate["candidate_answer"])
        print("answer:",answer)
        if " ".join(answer) in pd_candidate["candidate_answer"].apply(lambda x:" ".join(x)):
            print("++++++++++effective answer+++++++++++")

            # 根据slide window计算最好的answer
            for index, row in pd_candidate.iterrows():
                candidate = row['candidate_answer']  # 每一个row就是一个候选答案，其实candidate已经分好了词
                candidate.extend(question_words)  # 可能这里和python2有区别，python3中extend直接在candidate基础上进行扩展，没有返回值
                S = set(candidate)  # question和answer word的并集
                sw = []
                for j in range(1, len(passage_words) + 1):  # 对于passage中的每个词
                    temp = 0
                    for w in range(1, len(S) + 1):  # 对于并集中的每个词
                        word = passage_words[j + w - 1:]  # 可能已经到头了，下面应该去第一个词即word[0]
                        if len(word) > 0 and word[0] in S:
                            temp += __IC(passage_words, word[0])
                        else:
                            temp += 0
                    sw.append(temp)
                pd_candidate['slide_window'][index] = max(sw)  # 每个答案的slide window值

            pd_candidate = pd_candidate[
                pd_candidate['slide_window'] == pd_candidate['slide_window'].max(0)]  # 最大值所在的answers就是最终的answers

    print("****** max sliding window value ******")
    #print(pd_candidate)
    return pd_candidate


def main():
    #函数调用
    reader = SquadReader()
    filename = dp.DEV_DATA
    reader.load_dataset(filename)
    reader.process_data()
    pd_data = reader.pd_data
    answer = []
    print("pd_data len:",len(pd_data))
    for index,row in pd_data.iterrows():  #这里迭代的是每篇文章以及对应的question，传入的是pd_data的一行
        print("now index num:",index)
        begin = time.time()  #起始
        pd_answer = slide_window(row)
        if len(pd_answer) > 1:
            print("--------final multi answer-------")
        else:
            print("--------final answer--------")
        print("question:", row['question'])
        print("candidate answer:", list(pd_answer['candidate_answer']))
        print("true answer:", row['answer'])
        end = time.time()
        print("total time calculate:",end-begin," index num:",index)
        answer.append(list(pd_answer['candidate_answer']))

    print(answer)


if __name__ == '__main__':
    # input1 = "my name is zhu chauncy what your name is."
    # input2 = "your name is not zhu chauncy your name zhu"
    # n = 2
    # input1 = nltk.word_tokenize(input1)
    # input1 = [w for w in input1 if (w not in stopwords.words('english'))]
    # output1 = __getNgrams(input1, n)
    # print(output1)
    #
    # input2 = nltk.word_tokenize(input2)
    # input2 = [w for w in input2 if (w not in stopwords.words('english'))]
    # output2 = __getNgrams(input2, n)
    # print(output2)
    #
    # #output3 = [list(filter(lambda x: x in output1, sublist)) for sublist in output2]
    # output3 = set(output1).intersection(output2)
    # print(output3)
    begin = time.time()
    main()
    end = time.time()
    print("total time calculate:",end-begin)
