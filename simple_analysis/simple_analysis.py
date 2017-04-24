# coding=gbk
import utils.data_path as dp
import json
"""
    数据导入，主要是json数据，train是训练集，dev是测试集
"""
#SOURCE_DATA = dp.TRAIN_DATA  #训练数据
SOURCE_DATA = dp.DEV_DATA  #测试数据

"""
    数据结构
    {'data':
        [{ ----有多个
            'title':string,
            'paragraphs':[{ ----有多个
                'context':string,
                'qas':[{ ---对于一个context有多个question and answer
                    'id':string,
                    'question':string,  ---训练集question和answer一共有87599对，测试集中有10570对
                    'answers':[{  ---对于训练集中的一个question好像只有一个？经过实验发现每个问题对应一个答案，而对于测试集中的question，可能会有多个答案
                        'answer_start':int,
                        'text':string}]}]}]}],
    'version':'1.1'}

"""

f = open(SOURCE_DATA, encoding='utf-8')
total_data = json.load(f)
data = total_data['data']   #数据，data length:442(paragraphs)
version = total_data['version']  #版本号
multi_answer = [0]*6
answer_in_question_num = 0 #answer在question中出现的个数
question_answer_num = 0
for one in data:
    paragraphs = one['paragraphs']  #段落
    print("paragraphs lenght:",len(paragraphs))
    for two in paragraphs:
        context_list = two['context'].split(".")  #原文，分成不同的句子
        qas = two['qas']  #问答
        print("qas lenght:", len(qas))
        for three in qas:
            question_answer_num += 1
            question = three['question']
            answers = {i['text']: i['answer_start'] for i in three['answers']}  # Take only unique answers

            # for answer in answers:
            #     answer_start = answer['answer_start']
            #     text = answer['text']
            #     if text in question:
            #         answer_in_question_num += 1

                # sentence_text_num = 0   #统计answer在context中出现的句子个数
                # for context in context_list:
                #     if text in context:
                #         sentence_text_num += 1
                # if sentence_text_num == 0:
                #     print("question is:",question,"answer is:",text)
                # print("answer in multi sentence:",sentence_text_num)

            for i in range(6):  #最多5个答案，依次个数[0, 4945, 4051, 1368, 166, 40, 0, 0]
                if len(answers) == i:
                    multi_answer[i] += 1
print("multi_answer:",multi_answer)
print("question answer number:",question_answer_num,"answer in question num:",answer_in_question_num)
print("data length:",len(data),"verson:",version)