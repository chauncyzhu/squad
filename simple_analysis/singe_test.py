# coding=gbk
"""
    һ������������ʹ�
"""
import os
import utils.data_path as dp
from nltk.parse import stanford
from nltk import word_tokenize
import candidate_answer.stanford_parser as sp


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

def single_test():
    # ����stanford��������,�˴���Ҫ�ֶ��޸ģ�jar����ַΪ���Ե�ַ��
    os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
    os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
    # ΪJAVAHOME���ӻ�������
    java_path = dp.JAVA_PATH
    os.environ['JAVAHOME'] = java_path
    PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)

    single_line = "the american football conference (afc) champion denver broncos defeated the national football conference (nfc) champion carolina panthers 24??0 to earn their third super bowl title."

    constituents_list = sp.getConstituents(PAESER, single_line)  # ע�⣬����ʵ����������Ƕ���б�����Ҫ����ѭ�����н���
    print(constituents_list)
    sentence_one = word_tokenize("the american football conference (afc) champion denver broncos defeated the national football conference (nfc) champion carolina panthers 24??0 to earn their third super bowl title.")
    sentence_two = word_tokenize("as this was the 50th super bowl, the league emphasized the ""golden anniversary"" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each super bowl game with roman numerals (under which the game would have been known as ""super bowl l""), so that the logo could prominently feature the arabic numerals 50.")

    question = word_tokenize("which nfl team represented the afc at super bowl 50?")

    answer = "denver" #����broncos

    sentence_one_ngram = __getNgrams(sentence_one,1)
    sentence_two_ngram = __getNgrams(sentence_two, 1)
    overlap = __overlap(sentence_one_ngram,sentence_two_ngram)

    print(overlap)

if __name__ == '__main__':
    single_test()