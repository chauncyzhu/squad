# coding=gbk
"""
    ʹ��stanford parser���н���
"""
import os
import utils.data_path as dp
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
    if node.height() > 2:
        # print("node:",node)
        # print("node leaves:",node.leaves())
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
    parse_line = parser.raw_parse(single_line)
    constituents = []
    print("constituency parse by stanford corenlp")
    for line in parse_line:
        #line.draw()
        for root in line:
            #print("parse tree:", root)
            for tree in root:
                __getLeaves(tree, constituents)
    return constituents

if __name__ == '__main__':
    pass
    # single_line = "super bowl 50 was an american football game to determine the champion of the national football league (nfl) for the 2015 season"
    # constituents_list = getConstituents(single_line)  #ע�⣬����ʵ����������Ƕ���б�����Ҫ����ѭ�����н���
    # for constituent in constituents_list:
    #     line = " ".join(constituent)
    #     print("line:",line)

# GUI
# line.draw()