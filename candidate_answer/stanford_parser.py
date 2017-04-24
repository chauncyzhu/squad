# coding=gbk
"""
    使用stanford parser进行解析
"""
import os
import utils.data_path as dp
from nltk.parse import stanford

#添加stanford环境变量,此处需要手动修改，jar包地址为绝对地址。
os.environ['STANFORD_PARSER'] = dp.STANFORD_PARSER
os.environ['STANFORD_MODELS'] = dp.STANFORD_MODELS
#为JAVAHOME添加环境变量
java_path = dp.JAVA_PATH
os.environ['JAVAHOME'] = java_path

#如果height>2，则递归调用方法

def __getLeaves(node,constituents):
    """
    获取所有的constituents，根节点对应的所有词以及子树对应的所有词
    :param node: 经过parse的tree
    :param constituents: 属于constituency parse
    :return: constituents--注意不包括单个词
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
    对single line进行解析
    :param parser: 传入parser
    :param single_line: string
    :return:
    """
    # 句法标注
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
    # constituents_list = getConstituents(single_line)  #注意，这里实际上是三层嵌套列表，需要两个循环进行解析
    # for constituent in constituents_list:
    #     line = " ".join(constituent)
    #     print("line:",line)

# GUI
# line.draw()