1.读取train-v1.1.json和dev-v1.1.json，转换成pandas dataframe并写入csv中
dataframe中包含：
passage：每篇文章
passage_words：每篇文章的分词，仅仅使用了word_tokenize，没有去停用词，list(string)
passage_length：文章的长度
sentences：通过split(".")，分成了一个个句子
question：问题，string
question_words：使用word_tokenize分词
question_id：问题的id，这个应该是唯一的
answer：正确的答案，dict{answer_text:answer_start}
answer_words：使用word_tokenize分词

2.关于dev.json数据
(1)当generate_candidate读到第275个时，'≠'无法和正确解析出来
(2)当generate_candidate读到第340个时，

3.slide window效果非常差，用了一千多条数据做测试，如果限制答案长度，则准确率仅为1%，如果不限制答案长度，则准确率为25%~30%

