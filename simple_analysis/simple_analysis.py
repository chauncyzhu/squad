# coding=gbk
import utils.data_path as dp
import json
"""
    ���ݵ��룬��Ҫ��json���ݣ�train��ѵ������dev�ǲ��Լ�
"""
#SOURCE_DATA = dp.TRAIN_DATA  #ѵ������
SOURCE_DATA = dp.DEV_DATA  #��������

"""
    ���ݽṹ
    {'data':
        [{ ----�ж��
            'title':string,
            'paragraphs':[{ ----�ж��
                'context':string,
                'qas':[{ ---����һ��context�ж��question and answer
                    'id':string,
                    'question':string,  ---ѵ����question��answerһ����87599�ԣ����Լ�����10570��
                    'answers':[{  ---����ѵ�����е�һ��question����ֻ��һ��������ʵ�鷢��ÿ�������Ӧһ���𰸣������ڲ��Լ��е�question�����ܻ��ж����
                        'answer_start':int,
                        'text':string}]}]}]}],
    'version':'1.1'}

"""

f = open(SOURCE_DATA, encoding='utf-8')
total_data = json.load(f)
data = total_data['data']   #���ݣ�data length:442(paragraphs)
version = total_data['version']  #�汾��
multi_answer = [0]*6
answer_in_question_num = 0 #answer��question�г��ֵĸ���
question_answer_num = 0
for one in data:
    paragraphs = one['paragraphs']  #����
    print("paragraphs lenght:",len(paragraphs))
    for two in paragraphs:
        context_list = two['context'].split(".")  #ԭ�ģ��ֳɲ�ͬ�ľ���
        qas = two['qas']  #�ʴ�
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

                # sentence_text_num = 0   #ͳ��answer��context�г��ֵľ��Ӹ���
                # for context in context_list:
                #     if text in context:
                #         sentence_text_num += 1
                # if sentence_text_num == 0:
                #     print("question is:",question,"answer is:",text)
                # print("answer in multi sentence:",sentence_text_num)

            for i in range(6):  #���5���𰸣����θ���[0, 4945, 4051, 1368, 166, 40, 0, 0]
                if len(answers) == i:
                    multi_answer[i] += 1
print("multi_answer:",multi_answer)
print("question answer number:",question_answer_num,"answer in question num:",answer_in_question_num)
print("data length:",len(data),"verson:",version)