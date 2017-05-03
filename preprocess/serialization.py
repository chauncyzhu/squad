# coding=gbk
"""
    将json序列化到csv中
"""
import json
import utils.data_path as dp
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords

def load_dataset(filename):
    """
    the SQuAD dataset is only 29MB no problem in replicating the documents.
    means each question-answer pair with a passage
    :param filename: file name of the dataset
    :return: self.pd_data = pd_data(all lower words)
    """
    pd_data = pd.DataFrame(columns=["passage","passage_id","question", "answer", 'question_id'])  # 其中answer是dict类型的

    dataset = json.load(open(filename))["data"]
    count = 0
    passage_id = 0
    for doc in dataset:
        for paragraph in doc["paragraphs"]:
            p = paragraph['context'].lower()  # 转成了小写
            p_id = passage_id   #passage_id 从
            passage_id += 1
            for question in paragraph['qas']:
                answers = {i['text'].lower(): i['answer_start'] for i in
                           question['answers']}  # Take only unique answers
                q = question['question'].lower()
                q_id = question['id']
                pd_data.loc[count] = [p, p_id, q, answers, q_id]
                count += 1
                # if count>2:
                #     break
                print("has read " + str(count) + " passage-question-answer pair")

    return pd_data


"""
对数据进行预处理，对象为pandas dataframe进行分词，暂时不要去除停用词，统一转成小写
passage：根据"."进行分割成sentences，passage总length、word_set
question：words set
answer：words set
"""
def process_data(pd_data):
    if len(pd_data) < 1:
        print("please load dataset.")
        return
    # 分词并去停用词
    def f(x):
        return [w for w in word_tokenize(x) if (w not in stopwords.words('english'))]
    pd_data["passage_words"] = pd_data["passage"].apply(f)
    pd_data["question_words"] = pd_data["question"].apply(f)
    pd_data["answer_words"] = pd_data["answer"].apply(lambda x: [f(key) for key in x.keys()])

    # passage分割成一个个句子，并去掉停用词
    def g(x):
        x = x.split(".")  #分成一个个句子
        all_sen = []
        for sen in x:
             all_sen.append(" ".join([w for w in word_tokenize(sen) if (w not in stopwords.words('english'))]))
        return all_sen
    pd_data["sentences"] = pd_data["passage"].apply(g)
    # passage的长度
    pd_data["passage_length"] = pd_data["passage"].apply(len)

if __name__ == '__main__':
    #filename = dp.TRAIN_DATA
    filename = dp.DEV_DATA
    pd_data = load_dataset(filename)
    #process_data(pd_data)
    print(pd_data)
    pd_data.to_csv(dp.DEV_PD,encoding="utf8")