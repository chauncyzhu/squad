# coding=gbk
"""
    �������ڷ���--�����Ķ�������candidate answer���ɷ�ʽ��δŪ���������д�����Ĵ���
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

#����stanford��������,�˴���Ҫ�ֶ��޸ģ�jar����ַΪ���Ե�ַ��
os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
#ΪJAVAHOME���ӻ�������
java_path = dp.JAVA_PATH
os.environ['JAVAHOME'] = java_path
PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)
N_GRAM = 2

def candidate_answer(sentences):
    """
    ����passage���ɺ�ѡ�𰸣����غ�ѡ���Լ����ڵľ���
    :param sentences: �����е�һ��������
    :return: ���� ���Լ����ڵľ��ӣ�pandas dataframe--candidate answer��Ӧ��sentence��Ӧ�ð���candidate answer
    """
    candidate_dict = {}
    print("--------begin find all candidate answers.---------")
    for sen in sentences:  #�����һ��list
        if sen:
            print("****** sentence in the passage ******")
            constituents = sp.getConstituents(PAESER,sen)  #ʵ���Ͻ�����ʱ��Ҳ�Ứ�ѱȽ϶��ʱ��
            if sen not in candidate_dict:
                candidate_dict[sen] = constituents

    pd_candidate = pd.DataFrame(columns=['candidate_answer','sentence'])
    count = 0
    for key,value in candidate_dict.items():
        for va in value:
            pd_candidate.loc[count] = [va,key.replace(" ".join(va),'')]  #ȥ��candidate answer����
            count += 1
    print("****** all candidate answer ******")
    print("--------end find all candidate answers.---------")
    return pd_candidate

def clean_data(data):
    """
    �������ݣ��ִʡ�ȥ��ͣ�ô�
    :param data: ���� string
    :return: data list
    """
    data = word_tokenize(data)
    data = [w for w in data if (w not in stopwords.words('english'))]
    return data

def __getNgrams(input, n):
    """
    ��ȡngramģ��
    :param input: list���Ѿ��ֺôʲ�ȥ����stopwords
    :param n: ngram�е�n
    :return: ����ngramģ��
    """
    output = [] # �����ֵ�
    for i in range(len(input)-n+1):
        ngramTemp = " ".join(input[i:i+n])  #.encode('utf-8')
        output.append(ngramTemp)
    return output

def __overlap(dict_one,dict_two):
    """
    ��������dict֮���overlap
    :param dict_one:�ʵ�1
    :param dict_two:�ʵ�2
    :return:��������֮��overlap��ֵ
    """
    return len(set(dict_one).intersection(dict_two))

def __maximal_overlap(pd_candidate,question):
    """
    ���ÿ����ѡanswer���ڵ�����overlap��sentence
    :param pd_candidate: dataframe ��ѡ�𰸣�������candidate_answer�Լ���Ӧ��sentence
    :param question: questionֻ��һ��
    :return: pd_candidate ��Ϊ��������ɸѡ�������Ҫreturn �����������overlap��candidate answer(bool��ʾ)
    """
    #print("question:",question)
    def f(x):
        # ����sentence��question��unigram/bigram(Ӧ���ǻ��ߵ���˼���������bigram) overlap
        question_bigram = __getNgrams(question, N_GRAM)
        sentence_bigram = __getNgrams(x, N_GRAM)
        overlap = __overlap(question_bigram, sentence_bigram)  # ��������֮���overlap
        return overlap

    pd_candidate['overlap'] = pd_candidate['sentence'].apply(f)
    pd_candidate = pd_candidate[pd_candidate['overlap'] == pd_candidate['overlap'].max(0)]  #��ȡ����maximal overlap

    print("****** maximal overlap candidate answer ******")
    #print(pd_candidate)
    return pd_candidate

def __IC(passage,word):
    Cw = passage.count(word)  #passage����word�ĸ���
    return math.log(1+1/float(Cw))


def slide_window(row):
    """
    ͨ��slide windowѡ������ʵ�answer
    :param row: pd_data��һ�У�������passage��passage_words��sentences��question��question_words
    :return:final_answer �������յĴ� pd_candidate pandas dataframe
    """
    passage = row['passage']  #����Ψһ
    passage_words = row['passage_words']
    question = row['question']  #����Ψһ
    question_words = row['question_words']

    #�ҳ����к�ѡ��
    pd_candidate = candidate_answer(row['sentences'])

    #����unigram/bigram�ҳ�����sentence��question����overlap��Ӧ��candidate answer
    pd_candidate = __maximal_overlap(pd_candidate,question)  #�ҵ�ÿ��answer���overlap���ڵ�sentence����ʱpd_candidate�Ѿ�����ɸѡ

    pd_candidate['slide_window'] = list(range(len(pd_candidate)))

    for answer in row["answer_words"]:
        print("pd_candidate candidate_answer:",pd_candidate["candidate_answer"])
        print("answer:",answer)
        if " ".join(answer) in pd_candidate["candidate_answer"].apply(lambda x:" ".join(x)):
            print("++++++++++effective answer+++++++++++")

            # ����slide window������õ�answer
            for index, row in pd_candidate.iterrows():
                candidate = row['candidate_answer']  # ÿһ��row����һ����ѡ�𰸣���ʵcandidate�Ѿ��ֺ��˴�
                candidate.extend(question_words)  # ���������python2������python3��extendֱ����candidate�����Ͻ�����չ��û�з���ֵ
                S = set(candidate)  # question��answer word�Ĳ���
                sw = []
                for j in range(1, len(passage_words) + 1):  # ����passage�е�ÿ����
                    temp = 0
                    for w in range(1, len(S) + 1):  # ���ڲ����е�ÿ����
                        word = passage_words[j + w - 1:]  # �����Ѿ���ͷ�ˣ�����Ӧ��ȥ��һ���ʼ�word[0]
                        if len(word) > 0 and word[0] in S:
                            temp += __IC(passage_words, word[0])
                        else:
                            temp += 0
                    sw.append(temp)
                pd_candidate['slide_window'][index] = max(sw)  # ÿ���𰸵�slide windowֵ

            pd_candidate = pd_candidate[
                pd_candidate['slide_window'] == pd_candidate['slide_window'].max(0)]  # ���ֵ���ڵ�answers�������յ�answers

    print("****** max sliding window value ******")
    #print(pd_candidate)
    return pd_candidate


def main():
    #��������
    reader = SquadReader()
    filename = dp.DEV_DATA
    reader.load_dataset(filename)
    reader.process_data()
    pd_data = reader.pd_data
    answer = []
    print("pd_data len:",len(pd_data))
    for index,row in pd_data.iterrows():  #�����������ÿƪ�����Լ���Ӧ��question���������pd_data��һ��
        print("now index num:",index)
        begin = time.time()  #��ʼ
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