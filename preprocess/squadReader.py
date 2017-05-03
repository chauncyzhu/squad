# coding=gbk
"""
    squad数据读取类：从csv中读取数据
    可参考：https://github.com/hadyelsahar/SQuAD/blob/master/utils/datareader.py
"""
import json
import utils.data_path as dp
import pandas as pd
from nltk import word_tokenize


class SquadReader:
    def load_simple_dataset(self,filename):
        """
        直接加载经过序列化后的数据集，columns=["passage","passage_id","question", "answer", 'question_id']
        :param filename: 文件名
        :return: 
        """
        pd_data = pd.read_csv(filename,index_col=0)  #由于均为string所以不需要进行eval转换
        pd_data['answer'] = pd_data['answer'].apply(eval)  #还原，原先存储的是字典

        return pd_data

    def load_pd_dataset(self, filename):
        """
        the SQuAD dataset is only 29MB no problem in replicating the documents.
        means each question-answer pair with a passage
        :param filename: file name of the dataset
        :return: self.pd_data = pd_data(all lower words)
        """
        pd_data = pd.read_csv(filename,index_col=0)

        pd_data['answer'] = pd_data['answer'].apply(eval)  #还原
        pd_data['passage_words'] = pd_data['passage_words'].apply(eval)  # 还原
        pd_data['question_words'] = pd_data['question_words'].apply(eval)  # 还原
        pd_data['answer_words'] = pd_data['answer_words'].apply(eval)  # 还原
        pd_data['sentences'] = pd_data['sentences'].apply(eval)  # 还原
        return pd_data


    def load_candidate_dataset(self,filename):
        pd_data = pd.read_csv(filename,index_col=0)
        #print("pd data:",pd_data)
        pd_data['candidate_answer'] = pd_data['candidate_answer'].apply(eval)  #还原
        return pd_data


if __name__ == '__main__':
    squadReader = SquadReader()
    # filename = dp.DEV_PD
    # pd_data = squadReader.load_pd_dataset(filename)
    # print(pd_data)

    filename = dp.CANDIDATE_ANSWERS
    pd_data = squadReader.load_candidate_dataset(filename)
    print(pd_data)


