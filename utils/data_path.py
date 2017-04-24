# coding=gbk
"""
    ����·��
"""
ROOT_PATH = "C:/Users/chauncy/Desktop/SQuAD/"

TRAIN_DATA = ROOT_PATH + "train-v1.1.json"  #ѵ������
DEV_DATA = ROOT_PATH + "dev-v1.1.json" #dev

TRAIN_PD = ROOT_PATH + "train.csv"
DEV_PD = ROOT_PATH + "dev.csv"
CANDIDATE_ANSWERS = ROOT_PATH + "candidate_answers.csv"


# stanford nlp����
STANFORD_PATH = "D:/Coding/pycharm-professional/nltk/stanford_nlp/parser/"
JAVA_PATH = "C:/Program Files/Java/jdk1.8.0_121/bin/java.exe"

STANFORD_PARSER = STANFORD_PATH + "stanford-parser.jar"
STANFORD_MODELS = STANFORD_PATH + "stanford-parser-3.7.0-models.jar"
ENGLISHPCFG = STANFORD_PATH + "englishPCFG.ser.gz"