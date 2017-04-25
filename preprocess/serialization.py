# coding=gbk
"""
    ��json���л���csv��
"""
import json
import utils.data_path as dp
import pandas as pd
from nltk import word_tokenize

def load_dataset(filename):
    """
    the SQuAD dataset is only 29MB no problem in replicating the documents.
    means each question-answer pair with a passage
    :param filename: file name of the dataset
    :return: self.pd_data = pd_data(all lower words)
    """
    pd_data = pd.DataFrame(columns=["passage", "question", "answer", 'question_id'])  # ����answer��dict���͵�

    dataset = json.load(open(filename))["data"]
    count = 0
    for doc in dataset:
        for paragraph in doc["paragraphs"]:
            p = paragraph['context'].lower()  # ת����Сд
            for question in paragraph['qas']:
                answers = {i['text'].lower(): i['answer_start'] for i in
                           question['answers']}  # Take only unique answers
                q = question['question'].lower()
                q_id = question['id']
                pd_data.loc[count] = [p, q, answers, q_id]
                count += 1
                # if count>2:
                #     break
                print("has read " + str(count) + " passage-question-answer pair")

    return pd_data


"""
�����ݽ���Ԥ����������Ϊpandas dataframe���зִʣ���ʱ��Ҫȥ��ͣ�ôʣ�ͳһת��Сд
passage������"."���зָ��sentences��passage��length��word_set
question��words set
answer��words set
"""
def process_data(pd_data):
    if len(pd_data) < 1:
        print("please load dataset.")
        return
    # �ִ�
    pd_data["passage_words"] = pd_data["passage"].apply(word_tokenize)
    pd_data["question_words"] = pd_data["question"].apply(word_tokenize)
    pd_data["answer_words"] = pd_data["answer"].apply(lambda x: [word_tokenize(key) for key in x.keys()])

    # passage�ָ��һ��������
    pd_data["sentences"] = pd_data["passage"].apply(lambda x: x.split("."))
    # passage�ĳ���
    pd_data["passage_length"] = pd_data["passage"].apply(len)

if __name__ == '__main__':
    filename = dp.DEV_DATA
    pd_data = load_dataset(filename)
    process_data(pd_data)
    print(pd_data)
    pd_data.to_csv(dp.DEV_PD,encoding="utf8")