# coding=gbk
"""
    生成候选答案，由于这个步骤会花费较多的时间因此单独进行
"""
import os
import time

import pandas as pd
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

def candidate_answer(sentences,pd_candidate):
    """
    根据passage生成候选答案，返回候选答案以及所在的句子
    :param sentences: 文章中的一个个句子
    :param pd_candidate: 候选答案，传引
    :return: 传引 答案以及所在的句子，pandas dataframe--candidate answer对应的sentence不应该包括candidate answer
    """
    candidate_dict = {}
    print("--------begin find all candidate answers.---------")
    for sen in sentences:  # 这个是一个list
        if sen:
            constituents = sp.getConstituents(PAESER, sen)  # 实际上解析的时候也会花费比较多的时间
            if sen not in candidate_dict:
                candidate_dict[sen] = constituents

    temp_candidate = pd.DataFrame(columns=['candidate_answer', 'sentence'])
    count = 0
    for key, value in candidate_dict.items():
        for va in value:
            temp_candidate.loc[count] = [va, key.replace(" ".join(va), '')]  # 去掉candidate answer本身
            count += 1
    pd_candidate = pd.concat([pd_candidate,temp_candidate],ignore_index=True)
    print("--------end find all candidate answers.---------")
    return pd_candidate

def main():
    # 函数调用
    reader = SquadReader()
    filename = dp.DEV_DATA
    reader.load_dataset(filename)
    reader.process_data()
    pd_data = reader.pd_data
    answer = []
    pd_candidate = pd.DataFrame(columns=['candidate_answer', 'sentence','question_id'])
    stop_flag = 0
    for index, row in pd_data.iterrows():  # 这里迭代的是每篇文章以及对应的question，传入的是pd_data的一行
        print("now index num:", index)
        begin = time.time()  # 起始
        pd_candidate = candidate_answer(row['sentences'],pd_candidate)
        if stop_flag > 5:
            break
        stop_flag += 1
    print(pd_candidate)
    pd_candidate.to_csv(dp.CANDIDATE_ANSWERS)

if __name__ == '__main__':
    main()
