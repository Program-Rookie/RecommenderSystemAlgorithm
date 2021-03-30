import os
import math
import operator
#----------------------method1 手写-------------------------------
# 1、读取文档(s)
# 2、分词，英文直接用空格分词，中文是要麻烦些
# 3、去除停用词
# 4、计算词频
# 5、计算逆文档频率
# 6、计算tf-idf
def getDocumentsPath(filePath):
    '''
    读取文件夹下所有文件名
    :param filePath: 文件夹路径
    :return:
    '''
    arr = []
    for root, dirs, files in os.walk(filePath):
        for file in files:
            arr.append(os.path.join(root, file))
    return arr
# print(getDocumentsPath("./documents"))

def getDocumentsPath(path, filePaths):
    '''
    递归读取文件夹下所有文件名
    :param path: 文件夹路径
    :return:
    '''
    for file in os.listdir(path):
        filePath = os.path.join(path, file)
        if os.path.isdir(filePath):
            getDocumentsPath(filePath, filePaths)
        elif os.path.splitext(filePath)[1] == '.txt':
            filePaths.append(filePath)
# filePaths = []
# getDocumentsPath("./documents", filePaths)
# print(filePaths)

def readDocument(path):
    fr = open(path, encoding='utf-8')
    content = []
    for line in fr.readlines():
        content.append(line)
    return content

# print(readDocument("./documents/baidu_baike_mybatis.txt"))

def splitDocument(document):
    import jieba
    '''
    分词,
    :param document: readDocument读出来的list数据,一个元素是一行字符
    :return: 分词List
    '''
    allTxt = ''.join(document)
    return list(jieba.cut(allTxt, cut_all=False))

# print(splitDocument(readDocument("./documents/baidu_baike_mybatis.txt")))

def getStopWords(fileName):
    '''
    获取停用词列表, 这里用的停用词
    :return: 停用词列表
    '''
    stopWordList = []
    for line in readDocument(fileName):
        stopWordList.append(str(line).replace("\n", ''))
    return stopWordList

# print(getStopWords("./stopwords/baidu_stopwords.txt"))

def removeStopWords(stopWordList, originalWordList):
    '''
    去除停用词
    :param stopWordList: 停用词列表
    :param originalWordList: 分词
    :return:
    '''
    wordListWithoutStopWords = []
    for word in originalWordList:
        if str(word) in stopWordList:
            continue
        else:
            wordListWithoutStopWords.append(str(word))
    return wordListWithoutStopWords

def corpus(filePathList, stopWordList):
    '''
    建立语料库(分词,去除停用词后的形式)
    :param filePathList: 文件路径
    :param stopWordList:
    :return:
    '''
    allList = []
    for filePath in filePathList:
        wordListWithoutStopWords = removeStopWords(stopWordList, splitDocument(readDocument(filePath)))
        allList.append(wordListWithoutStopWords)
    return allList

# filePaths = []
# getDocumentsPath('./documents', filePaths)
# print(corpus(filePaths, getStopWords("./stopwords")))

def __calculateWordFrequency(wordListWithoutStopWords):
    '''
    计算词频
    :param wordListWithoutStopWords: 分词、去除停用词后的单个文档词列表
    :return: 词频字典
    '''
    wordFrequency = {}
    for word in wordListWithoutStopWords:
        if str(word) in wordFrequency.keys():
            wordFrequency[str(word)] = wordFrequency[str(word)] + 1
        else:
            wordFrequency[str(word)] = 1
    return wordFrequency

def __calculateWordInFileCount(word, corpusList):
    '''
    计算逆文档词频inverse document frequency
    :param word: 单词
    :param corpusList:语料库 list
    :return:
    '''
    count = 0
    for documentProcessedWordList in corpusList:
        for documentWord in documentProcessedWordList:
            if word in set(documentWord):
                count = count + 1
            else:
                continue
    return count

def tf_idf(documentProcessedWordList, corpusList):
    '''

    :param documentProcessedWordList:  分词去除停用词后的单个文章中的词列表
    :param corpusList: 分词、去除停用词后的语料库
    :return:
    '''
    outdic = {}
    dic = __calculateWordFrequency(documentProcessedWordList)
    for word in documentProcessedWordList:
        # 计算tf
        tf = dic[str(word)]/len(documentProcessedWordList)
        idf = math.log(len(corpusList)/(__calculateWordInFileCount(word, corpusList) + 1))
        tfidf = tf * idf
        outdic[str(word)] = tfidf
    orderdic = sorted(outdic.items(), key=operator.itemgetter(1), reverse=True) # 由高到低排序
    return orderdic

def dic2string(dic):
    out = ''
    for i in dic:
        out = out + str(i) + '\n'
    return out
if __name__ == '__main__':
    pass
    filePathList = []
    getDocumentsPath("./documents", filePathList)
    # print(filePathList)
    # 这里下载的几个停用词列表好像都不咋的
    stopWordList = getStopWords("./stopwords/scu_stopwords.txt")
    # print(stopWordList)
    corpusList = corpus(filePathList, stopWordList)
    # print(corpusList)
    out = ''
    for filePath in filePathList:
        documentProcessedWordList = removeStopWords(stopWordList, splitDocument(readDocument(filePath)))
        tfidfdic = tf_idf(documentProcessedWordList, corpusList)
        out = out + dic2string(tfidfdic)

    print(out)
#----------------------method2 导包-------------------------------
