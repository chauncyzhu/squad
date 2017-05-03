# coding=gbk
"""
    ʹ��stanford parser���н���
"""
import os
import numpy as np
import utils.data_path as dp
import nltk.tree
from nltk import word_tokenize
from nltk.parse import stanford

#����stanford��������,�˴���Ҫ�ֶ��޸ģ�jar����ַΪ���Ե�ַ��
os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
#ΪJAVAHOME���ӻ�������
java_path = dp.JAVA_PATH
os.environ['JAVAHOME'] = java_path

#���height>2����ݹ���÷���

def __getLeaves(node,constituents):
    """
    ��ȡ���е�constituents�����ڵ��Ӧ�����д��Լ�������Ӧ�����д�
    :param node: ����parse��tree
    :param constituents: ����constituency parse
    :return: constituents--ע�ⲻ����������
    """
    if node.height() > 2 and (node.label() == 'NP' or 'VP'):
        # print("node:",node)
        # print("node label:",node.label())
        constituents.append(node.leaves())
        for sub_node in node:
            __getLeaves(sub_node, constituents)
    else:
        return constituents

def getConstituents(parser,single_line):
    """
    ��single line���н���
    :param parser: ����parser
    :param single_line: string
    :return:
    """
    # �䷨��ע
    # what impact did the high school education movement have on the presence of skilled workers?
    # during the mass high school education movement from 1910-1940, there was an increase in skilled workers
    single_line = single_line.replace(" ��",'')   #ע��������һ���ӣ�'��'�޷�����ȷ��������
    parse_line = parser.raw_parse(single_line)
    constituents = []
    print("constituency parse by stanford corenlp")
    for line in parse_line:
        line.draw()
        for root in line:
            #print("parse tree:", root)
            for tree in root:
                __getLeaves(tree, constituents)
    token_words = word_tokenize(single_line)   #�Ƿ���ϱ���
    constituents.extend([list(i) for i in np.array(token_words).reshape(len(token_words),1)])
    return constituents

if __name__ == '__main__':
    # ����stanford��������,�˴���Ҫ�ֶ��޸ģ�jar����ַΪ���Ե�ַ��
    os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
    os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
    # ΪJAVAHOME���ӻ�������
    java_path = dp.JAVA_PATH
    os.environ['JAVAHOME'] = java_path
    PAESER = stanford.StanfordParser(model_path=dp.ENGLISHPCFG)

    single_line = "during the mass high school education movement from 1910-1940, there was an increase in skilled workers"

    constituents_list = getConstituents(PAESER,single_line)  #ע�⣬����ʵ����������Ƕ���б�����Ҫ����ѭ�����н���
    print(constituents_list)
    # for constituent in constituents_list:
    #     line = " ".join(constituent)
    #     print("line:",line)

# GUI
# line.draw()