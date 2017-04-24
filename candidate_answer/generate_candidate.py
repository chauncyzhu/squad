# coding=gbk
"""
    ���ɺ�ѡ�𰸣������������Ứ�ѽ϶��ʱ����˵�������
"""
import os
import time

import pandas as pd
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

def candidate_answer(sentences,pd_candidate):
    """
    ����passage���ɺ�ѡ�𰸣����غ�ѡ���Լ����ڵľ���
    :param sentences: �����е�һ��������
    :param pd_candidate: ��ѡ�𰸣�����
    :return: ���� ���Լ����ڵľ��ӣ�pandas dataframe--candidate answer��Ӧ��sentence��Ӧ�ð���candidate answer
    """
    candidate_dict = {}
    print("--------begin find all candidate answers.---------")
    for sen in sentences:  # �����һ��list
        if sen:
            constituents = sp.getConstituents(PAESER, sen)  # ʵ���Ͻ�����ʱ��Ҳ�Ứ�ѱȽ϶��ʱ��
            if sen not in candidate_dict:
                candidate_dict[sen] = constituents

    temp_candidate = pd.DataFrame(columns=['candidate_answer', 'sentence'])
    count = 0
    for key, value in candidate_dict.items():
        for va in value:
            temp_candidate.loc[count] = [va, key.replace(" ".join(va), '')]  # ȥ��candidate answer����
            count += 1
    pd_candidate = pd.concat([pd_candidate,temp_candidate],ignore_index=True)
    print("--------end find all candidate answers.---------")
    return pd_candidate

def main():
    # ��������
    reader = SquadReader()
    filename = dp.DEV_DATA
    reader.load_dataset(filename)
    reader.process_data()
    pd_data = reader.pd_data
    answer = []
    pd_candidate = pd.DataFrame(columns=['candidate_answer', 'sentence','question_id'])
    stop_flag = 0
    for index, row in pd_data.iterrows():  # �����������ÿƪ�����Լ���Ӧ��question���������pd_data��һ��
        print("now index num:", index)
        begin = time.time()  # ��ʼ
        pd_candidate = candidate_answer(row['sentences'],pd_candidate)
        if stop_flag > 5:
            break
        stop_flag += 1
    print(pd_candidate)
    pd_candidate.to_csv(dp.CANDIDATE_ANSWERS)

if __name__ == '__main__':
    main()