import jieba
list = ["我是一只小小小小鸟", "怎么也飞不出，花花的世界"]
content = ' '.join(list)
# 分词
seg1 = jieba.cut(content, cut_all=False) # 默认模式
print(' '.join(seg1))
seg2 = jieba.cut(content, cut_all=True) # 全模式
print(' '.join(seg2))
seg3 = jieba.cut_for_search(content) # 搜索引擎模式
print(' '.join(seg3))
# 词性标注
import jieba.posseg as pseg
words = pseg.cut(content)
for word, flag in words:
    print('%s %s' %(word, flag))
# 关键字抽取
import jieba.analyse
# 基于tf-idf
keyword = jieba.analyse.extract_tags(content)
print(keyword)
keywords = jieba.analyse.extract_tags(content, topK=5, withWeight=True)
print(keywords)
# 基于text rank
keyword = jieba.analyse.textrank(content)
print(keyword)
keywords = jieba.analyse.textrank(content, topK=30, withWeight=True)
print(keywords)

# jieba分词词性标记含义, https://blog.csdn.net/huludan/article/details/52727298