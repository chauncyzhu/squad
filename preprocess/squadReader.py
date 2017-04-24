# coding=gbk
"""
    squad数据读取类：读取数据，并转换为训练集和测试集
    可参考：https://github.com/hadyelsahar/SQuAD/blob/master/utils/datareader.py
"""
import json
import utils.data_path as dp
import pandas as pd
from nltk import word_tokenize


class SquadReader:
    def __init__(self):
        self.pd_data = None

    def load_dataset(self, filename):
        """
        the SQuAD dataset is only 29MB no problem in replicating the documents.
        means each question-answer pair with a passage
        :param filename: file name of the dataset
        :return: self.pd_data = pd_data(all lower words)
        """
        pd_data = pd.DataFrame(columns=["passage","question","answer",'question_id'])  #其中answer是dict类型的

        dataset = json.load(open(filename))["data"]
        count = 0
        for doc in dataset:
            for paragraph in doc["paragraphs"]:
                p = paragraph['context'].lower()  #转成了小写
                for question in paragraph['qas']:
                    answers = {i['text'].lower(): i['answer_start'] for i in question['answers']}  # Take only unique answers
                    q = question['question'].lower()
                    q_id = question['id']
                    pd_data.loc[count] = [p,q,answers,q_id]
                    count += 1
                    # if count>2:
                    #     break
                    print("has read "+str(count)+" passage-question-answer pair")

        self.pd_data = pd_data

    """
    对数据进行预处理，对象为pandas dataframe进行分词，暂时不要去除停用词，统一转成小写
    passage：根据"."进行分割成sentences，passage总length、word_set
    question：words set
    answer：words set
    """
    def process_data(self):
        if len(self.pd_data) < 1:
            print("please load dataset.")
            return
        #分词
        self.pd_data["passage_words"] = self.pd_data["passage"].apply(word_tokenize)
        self.pd_data["question_words"] = self.pd_data["question"].apply(word_tokenize)
        self.pd_data["answer_words"] = self.pd_data["answer"].apply(lambda x:[word_tokenize(key) for key in x.keys()])

        #passage分割成一个个句子
        self.pd_data["sentences"] = self.pd_data["passage"].apply(lambda x:x.split("."))
        #passage的长度
        self.pd_data["passage_length"] = self.pd_data["passage"].apply(len)


if __name__ == '__main__':
    squadReader = SquadReader()
    filename = dp.DEV_DATA
    squadReader.load_dataset(filename)
    squadReader.process_data()
    pd_data = squadReader.pd_data
    print(pd_data)
    pd_data.to_csv(dp.DEV_PD)
