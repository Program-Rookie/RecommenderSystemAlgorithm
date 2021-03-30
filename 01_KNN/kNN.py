from numpy import *
import operator
def createDataSet():
    '''
    生成一个数据集
    :return:
    '''
    group = array([[1.0,1.1],
                   [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def file2matrix(fileName):
    '''
    加载数据集
    :return:
    '''
    fr = open(fileName)
    arrayLines = fr.readlines() # arrayLines 是一个list object
    numberOfLines = len(arrayLines)
    # 返回的dataSet与labels
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def autoNorm(dataSet):
    '''
    归一化特征
    :param dataSet:
    :return:
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1)) # tile复制
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 可视化

def classify(sample, dataSet, labels, k = 3):
    '''
    分类
    :param sample: 待分类向量
    :param dataSet: 训练样本集
    :param k: 近邻个数
    :return: 分类
    '''
    # 计算距离，欧氏距离
    dataSize = dataSet.shape[0]
    diffMat = tile(sample, (dataSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort() # argsort 按值升序排列下标
    print(sortedDistIndicies)
    # 选择距离最小的k个点， 统计分类
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 排序分类, 按从大到小
    sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1), reverse = True) # operator.itemgetter(1)
    return sortedClassCount[0][0]

def datingClassTest():
    '''
    测试
    :return:
    '''
    hoRatio = 0.30
    datingDataMat, datingLabels = file2matrix('datingTestSet')
    normMat, ranges, minVals = autoNorm(datingDataMat)

    m = normMat.shape[0]
    # 测试集占一部分
    numTestVecs = int(m * hoRatio)
    print(numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels, 3)
        print("the classifer came back with: %d, the real answer is: %d"%(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
           errorCount += 1.0
    print("the total error rate is: %f"%(errorCount / float(numTestVecs)))

def img2vector(fileName):
    '''
    读取单个32*32的图片
    :param fileName:
    :return:
    '''
    fr = open(fileName)
    returnVect = zeros(1, 32 * 32)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(line[j])
    return returnVect

if __name__ == '__main__':
    # group, labels = createDataSet()
    # print(classify([0, 0], group, labels, 3))

    # datingDataMat, datingLabels = file2matrix('datingTestSet')
    # print(datingDataMat)
    # print(datingLabels)

    # import matplotlib
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1],
    #            datingDataMat[:, 2],
    #            15 * array(datingLabels),
    #            15 * array(datingLabels)) # x y size color
    # plt.show()

    # normDataSet, ranges, minVals = autoNorm(datingDataMat)
    # print(normDataSet)
    # print(ranges)
    # print(minVals)

    datingClassTest()
