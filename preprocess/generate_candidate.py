# coding=gbk
"""
    ���ɺ�ѡ�𰸣������������Ứ�ѽ϶��ʱ����˵�������
    ���ں�ѡ�𰸶�Ӧÿƪ���£����Ӧ��ֱ�Ӷ�ÿƪ����������Ӧ�Ĵ�
"""
import os
import time
import pandas as pd
from nltk import word_tokenize
from nltk.parse import stanford
import utils.data_path as dp
from preprocess import stanford_parser as sp
from preprocess.squadReader import SquadReader

#����stanford��������,�˴���Ҫ�ֶ��޸ģ�jar����ַΪ���Ե�ַ��
os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
#ΪJAVAHOME���ӻ�������
java_path = dp.JAVA_PATH
os.environ['JAVAHOME'] = java_path
PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)


def candidate_answer(passage,passage_id):
    """
    ����passage���ɺ�ѡ�𰸣����غ�ѡ���Լ����ڵľ���
    :param passage: ��ƪ���£�����ÿƪ���º�ѡ��Ӧ����һ����
    :param passage_id: ���µ�id
    :return: ���Լ����ڵľ��ӡ��Ƴ��𰸵ľ��ӣ�pandas dataframe--candidate answer��Ӧ��sentence��Ӧ�ð���candidate answer
    """
    sentences = passage.split(".")  #����"."���ָ����

    candidate_dict = {}
    print("--------begin find all candidate answers.---------")
    for sen in sentences:  # �����һ��list
        if sen:
            constituents = sp.getConstituents(PAESER, sen)  # ʵ���Ͻ�����ʱ��Ҳ�Ứ�ѱȽ϶��ʱ��
            if sen not in candidate_dict:
                candidate_dict[sen] = constituents

    temp_candidate = pd.DataFrame(columns=['passage_id','candidate_answer', 'sentence','sentence_extend_candidate'])
    count = 0
    for key, value in candidate_dict.items():  #����ÿһ����������
        for va in value:  #����ÿһ����ѡ������
            key_temp = word_tokenize(key)
            for i in va:  #�Ժ�ѡ�𰸵�ÿһ���ʶ���
                if i in key_temp:
                    key_temp.remove(i)
            temp_candidate.loc[count] = [passage_id,va, key," ".join(key_temp)]  # ȥ��candidate answer����
            count += 1
    print("--------end find all candidate answers.---------")
    return temp_candidate

def main():
    # ��������
    reader = SquadReader()
    filename = dp.DEV_PD   #csv����
    pd_data = reader.load_simple_dataset(filename)  #û�о����ִ�
    #pd_data = pd_data.head(10)  #�������ȡ10��������
    pd_candidate = pd.DataFrame(columns=['passage_id','candidate_answer', 'sentence','sentence_extend_candidate'])
    begin = time.time()  # ��ʼ
    pd_passage = pd_data[["passage","passage_id"]].drop_duplicates()   #ѡȡ���ظ���passage
    #ѡ�񲿷�����
    pd_passage = pd_passage[pd_passage["passage_id"]<100]

    print(pd_passage)
    for index, row in pd_passage.iterrows():  # �����������ÿƪ�����Լ���Ӧ��question���������pd_data��һ��
        print("now passage num:", int(row['passage_id']))
        pd_candidate = pd.concat([pd_candidate,candidate_answer(row['passage'],row['passage_id'])],ignore_index=0)
    print(pd_candidate)
    pd_candidate.to_csv(dp.CANDIDATE_ANSWERS,encoding="utf8")
    end = time.time()
    print("total calculate time:",end-begin)

if __name__ == '__main__':
    main()