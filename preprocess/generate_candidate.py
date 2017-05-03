# coding=gbk
"""
    生成候选答案，由于这个步骤会花费较多的时间因此单独进行
    由于候选答案对应每篇文章，因此应该直接对每篇文章生成相应的答案
"""
import os
import time
import pandas as pd
from nltk import word_tokenize
from nltk.parse import stanford
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


def candidate_answer(passage,passage_id):
    """
    根据passage生成候选答案，返回候选答案以及所在的句子
    :param passage: 整篇文章，对于每篇文章候选答案应该是一样的
    :param passage_id: 文章的id
    :return: 答案以及所在的句子、移除答案的句子，pandas dataframe--candidate answer对应的sentence不应该包括candidate answer
    """
    sentences = passage.split(".")  #根据"."来分割句子

    candidate_dict = {}
    print("--------begin find all candidate answers.---------")
    for sen in sentences:  # 这个是一个list
        if sen:
            constituents = sp.getConstituents(PAESER, sen)  # 实际上解析的时候也会花费比较多的时间
            if sen not in candidate_dict:
                candidate_dict[sen] = constituents

    temp_candidate = pd.DataFrame(columns=['passage_id','candidate_answer', 'sentence','sentence_extend_candidate'])
    count = 0
    for key, value in candidate_dict.items():  #对于每一个句子来讲
        for va in value:  #对于每一个候选答案来讲
            key_temp = word_tokenize(key)
            for i in va:  #对候选答案的每一个词而言
                if i in key_temp:
                    key_temp.remove(i)
            temp_candidate.loc[count] = [passage_id,va, key," ".join(key_temp)]  # 去掉candidate answer本身
            count += 1
    print("--------end find all candidate answers.---------")
    return temp_candidate

def main():
    # 函数调用
    reader = SquadReader()
    filename = dp.DEV_PD   #csv数据
    pd_data = reader.load_simple_dataset(filename)  #没有经过分词
    #pd_data = pd_data.head(10)  #这里仅仅取10个做测试
    pd_candidate = pd.DataFrame(columns=['passage_id','candidate_answer', 'sentence','sentence_extend_candidate'])
    begin = time.time()  # 起始
    pd_passage = pd_data[["passage","passage_id"]].drop_duplicates()   #选取无重复的passage
    #选择部分数据
    pd_passage = pd_passage[pd_passage["passage_id"]<100]

    print(pd_passage)
    for index, row in pd_passage.iterrows():  # 这里迭代的是每篇文章以及对应的question，传入的是pd_data的一行
        print("now passage num:", int(row['passage_id']))
        pd_candidate = pd.concat([pd_candidate,candidate_answer(row['passage'],row['passage_id'])],ignore_index=0)
    print(pd_candidate)
    pd_candidate.to_csv(dp.CANDIDATE_ANSWERS,encoding="utf8")
    end = time.time()
    print("total calculate time:",end-begin)

if __name__ == '__main__':
    main()
