# coding=gbk
"""
    滑动窗口分析--机器阅读，由于candidate answer生成方式还未弄懂，因此先写分析的代码
"""
import math
import os
import time
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.parse import stanford
from collections import defaultdict
import utils.data_path as dp
from preprocess import stanford_parser as sp
from preprocess.squadReader import SquadReader

#添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
#为JAVAHOME添加环境变量
java_path = dp.JAVA_PATH
os.environ['JAVAHOME'] = java_path
PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)
N_GRAM = 1

def __compute_counts(stories):
    """
    统计每个passage中出现词语的个数
    :param stories: list(passage)  未分词
    :return: 
    """
    counts = defaultdict(lambda: 0.0)
    for passage in stories:
        passage_words = word_tokenize(passage.lower())
        for token in passage_words:
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
    data = word_tokenize(data.lower())
    data = [w for w in data if (w not in stopwords.words('english'))]
    return data

def __getNgrams(input, n):
    """
    获取ngram模型
    :param input: list，已经分好词并去除了stopwords
    :param n: ngram中的n
    :return: 返回ngram模型
    """
    if not isinstance(type(input),list):
        return []
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
    :return:返回两个之间overlap的值，根据intersection取交集
    """
    return len(set(dict_one).intersection(dict_two))

def __maximal_overlap(pd_candidate_passage,question_words):
    """
    获得每个候选answer所在的最大的overlap的sentence
    :param pd_candidate_passage: dataframe 候选答案，'passage_id','candidate_answer', 'sentence','sentence_extend_candidate'
    :param question_words: 只有一个，分好了词
    :return: pd_candidate 因为最后进行了筛选，因此需要return 返回所有最大overlap的candidate answer(bool表示)
    """
    #print("question_words:",question_words)
    def f(x):
        # 计算sentence和question的unigram/bigram(应该是或者的意思，这里计算bigram) overlap
        question_bigram = __getNgrams(question_words, N_GRAM)
        sentence_bigram = __getNgrams(x, N_GRAM)
        overlap = __overlap(question_bigram, sentence_bigram)  # 两个句子之间的overlap
        return overlap

    pd_candidate_passage['overlap'] = pd_candidate_passage['sentence_extend_candidate'].apply(f)   #这个值对于每个question都不一样，但是是中间值，因此不需要重新分配变量
    pd_candidate_passage = pd_candidate_passage[pd_candidate_passage['overlap'] == pd_candidate_passage['overlap'].max(0)]  #获取最大的maximal overlap

    print("****** maximal overlap candidate answer ******")

    return pd_candidate_passage

def __IC(passage,word,begin,end):
    """
    计算TC值，但是由于论文没有讲清楚，因此这里关于passage[begin:end]只是在single_test中发现这样可以
    :param passage: 文章，已经分好了词
    :param word: 对应的词
    :param begin: 起始
    :param end: 终止
    :return: IC值
    """
    Cw = passage[begin:end].count(word)  #passage包含word的个数
    return math.log(1+1/float(Cw))


def slide_window(pd_row,pd_candidate_passage,icounts):
    """
    通过slide window选出最合适的answer
    :param row: pd_data的一行，包含了"passage","passage_id","question", "answer", "question_id"
    :param pd_candidate_passage: 每篇文章对应的所有答案，包含了'passage_id','candidate_answer', 'sentence','sentence_extend_candidate'
    :param icounts: 每个单词的inverse counts IC值
    :return:pd_candidate_passage 由于重新赋值，因此需要return
    """
    passage = pd_row['passage']  #都是唯一
    question = pd_row['question']  #string，都是唯一

    #对passage和question进行分词
    passage_words = word_tokenize(passage.lower())
    question_words = word_tokenize(question.lower())

    #根据unigram/bigram找出所有sentence(去除了candidate的words)和question最大的overlap对应的candidate answer
    pd_candidate_passage = __maximal_overlap(pd_candidate_passage,question_words)  #找到每个answer最大overlap所在的sentence，此时pd_candidate已经经过筛选

    pd_candidate_passage['slide_window'] = [float(0)]*len(pd_candidate_passage)

    # 根据slide window计算最好的answer
    for index, row in pd_candidate_passage.iterrows():  # 每一个row就是一个候选答案，其实candidate已经分好了词
        candidate = row['candidate_answer']  #去停用词，注意copy()
        #candidate.extend(question_words)  # python3中extend直接在candidate基础上进行扩展，没有返回值
        S = set(candidate + question_words)  # question和answer word的并集
        S_len = len(S)
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
        pd_candidate_passage['slide_window'][index] = max(sw)  # 每个答案的slide window值

    pd_candidate_passage = pd_candidate_passage[pd_candidate_passage['slide_window'] == pd_candidate_passage['slide_window'].max(0)]  # 最大值所在的answers就是最终的answers
    #print(pd_candidate_passage)

    print("****** max sliding window value ******")

    return pd_candidate_passage


def evaluation(pd_row,pd_candidate_passage):
    """
    对slide windows选出的答案进行分析，看是否正确，关于这个正确与否，只能进行粗略的估计，'passage_id','candidate_answer', 'sentence','sentence_extend_candidate'
    注意这个evaluation只是对单个数据进行评判
    :param pd_row: question-answer 每一行，包含了很正确答案，其中answer是一个字典
    :param pd_candidate_passage: 通过slide window选择出的答案，可能有多个，只有当选出的答案全部符合条件才算这次正确
    :return: 1表示正确，0表示错误
    """
    true_answer_num = 0
    for index,row in pd_candidate_passage.iterrows():
        right_flag = False
        for key, value in pd_row['answer'].items():  # 对于每一个answer，只要在候选答案中出现过，候选答案就算正确
            if key in " ".join(row['candidate_answer']):  #如果真正的答案在候选答案的内部
                right_flag = True
                print("candidate answer:",row['candidate_answer'],"correct answer:",key)
                break
        if right_flag == True:  #如果正确答案都会在候选答案，说明该答案算正确
            true_answer_num += 1

    if true_answer_num == len(pd_candidate_passage):  #当所有答案正确的时候，才能说该问题找到了候选答案
        return 1
    else:
        return 0


def main():
    #函数调用
    reader = SquadReader()
    dev_csv = dp.DEV_PD
    candidate_csv = dp.CANDIDATE_ANSWERS
    evaluation_txt = dp.EVALUATION_TXT
    begin_begin = time.time()  # 起始

    pd_candidate = reader.load_candidate_dataset(candidate_csv)  #所有候选答案
    pd_data = reader.load_simple_dataset(dev_csv)  #所有数据，都是字符串，如果需要token则在程序中写出
    evaluation_file = open(evaluation_txt,'w')

    print("pd data len:",len(pd_data)," passage len:",len(pd_candidate))

    #pd_data = pd_data.loc[70:]  #控制pd data的个数
    icounts = __compute_inverse_counts(list(pd_data['passage']))
    precision, total_num = 0,0  #正确的个数
    for index,row in pd_data.iterrows():  #这里迭代的是每篇文章以及对应的question，传入的是pd_data的一行
        if row['passage_id'] in list(pd_candidate['passage_id']):  #如果能找到相应的passage
            print("now index num:", index," passage id:",int(row['passage_id']),"total num:",total_num)
            pd_candidate_passage = pd_candidate[pd_candidate['passage_id'] == row['passage_id']].copy()  # 找出每篇文章对应的question的所有候选答案，注意这里copy，防止传引
            begin = time.time()  # 起始
            pd_candidate_passage = slide_window(row, pd_candidate_passage,icounts)  #传引，只是对pd_candidate_passage进行操作
            precision += evaluation(row, pd_candidate_passage)
            end = time.time()
            print("total time calculate:", end - begin, " index num:", index, "total num:",total_num,"now precision num:",precision)
            total_num += 1
    #正确率
    print("precision num:",precision," final precision rate:",float(precision)/total_num)
    end_end = time.time()
    evaluation_file.write("precision num:"+str(precision)+"\ntotal num:"+str(total_num)+"\nfinal precision rate:"+str(float(precision)/total_num)+"\ntotal time calculate:"+str(end_end - begin_begin))

if __name__ == '__main__':
    begin = time.time()
    main()
    end = time.time()
    print("total time calculate:",end-begin)
