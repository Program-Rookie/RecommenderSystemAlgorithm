import pandas as pd
import numpy as np
# ItemCF的具体步骤
# 1、基于历史数据，构建以用户为行坐标，物品为列坐标的m×n的共现矩阵
# 2、计算共现矩阵两两列向量间的相似性，构建n×n的物品相似度矩阵
# 3、获得用户历史行为数据中的正反馈物品列表
# 4、利用物品相似度矩阵，针对目标用户历史行为中的正反馈物品，找出相似的topK个物品，组成相似物品集合
# 5、对相似物品集合中的物品，利用相似度分值进行排序，生成最终的推荐列表
# 参考链接：https://www.jianshu.com/p/a59ff0dc22a3
#           https://blog.csdn.net/qq_27575895/article/details/90410007

# 读取movieLens 1M
rnames = ["UserID", "MovieID", "Rating", "Timestamp"]
ratings = pd.read_table("../data/ml-1m/ratings.dat", sep="::", names=rnames, engine='python')
# print(ratings[:5])
#    UserID  MovieID  Rating  Timestamp
# 0       1     1193       5  978300760
# 1       1      661       3  978302109
# 2       1      914       3  978301968
# 3       1     3408       4  978300275
# 4       1     2355       5  978824291

# ratings_1 = ratings[['UserID', "MovieID", "Rating"]]
# print(ratings_1[:5])
# 数据量太大，抽取一部分来实验
df = ratings.sample(n=20, axis= 0, random_state=1)
# df = ratings.sample(n=20, axis= 0, random_state= 3)
# print(df)

def getUserItemMatrix(df, userColName, itemColName, ratingColName):
    '''
        获取共现矩阵
    :param df:
    :param userColName:
    :param itemColName:
    :param ratingColName:
    :return:
    '''
    users = df[userColName].unique()
    userCount = users.shape[0]
    # print(userCount)
    items = df[itemColName].unique()
    movieCount = items.shape[0]
    # print(movieCount)
    matrix = pd.DataFrame(np.zeros((userCount, movieCount)), index=users, columns=items)
    for i in df.itertuples():
        matrix.loc[getattr(i, userColName), getattr(i, itemColName)] = getattr(i, ratingColName)
    # matrix.to_csv("user_item_matrix.csv")
    # print(matrix[:5])
    # print(matrix.shape)
    return matrix

def cosineSimilarity(item1Vector, item2Vector):
    '''
    余弦相似度计算方法
    :param item1Vector:
    :param item2Vector:
    :return:
    '''
    assert item1Vector.shape == item2Vector.shape
    item1VectorNorm = np.linalg.norm(item1Vector, ord=2)
    item2VectorNorm = np.linalg.norm(item2Vector, ord=2)
    if item1VectorNorm is 0 or item2VectorNorm is 0:
        return 0
    if np.equal(item1Vector, item2Vector).all():
        return 1
    return np.dot(item1Vector, item2Vector) / (item1VectorNorm * item2VectorNorm)

def createItemSimilarityMatrix(userItemMatrix, itemNameList):
    '''
    构建物品相似度矩阵
    :param userItemMatrix: 物品集合
    :param itemLen: 总物品数
    :param itemNameList: 物品总集合（无重复）
    :return: 返回物品相似度矩阵，此处实际返回为DataFrame类型
    '''
    itemLen = len(itemNameList)
    itemSimilarityMatrix = pd.DataFrame(np.zeros((itemLen, itemLen)), index=itemNameList, columns=itemNameList)
    for i in range(len(itemNameList)):
        for j in range(len(itemNameList) - i):
            item1 = itemNameList[i]
            item2 = itemNameList[j + i]
            # print("item1:", item1)
            # print("item2:", item2)
            itemSimilarityMatrix.loc[item1, item2] = cosineSimilarity(userItemMatrix[item1], userItemMatrix[item2])
            itemSimilarityMatrix.loc[item2, item1] = itemSimilarityMatrix.loc[item1, item2]
    return itemSimilarityMatrix

def getItemCF(itemSimilarityMatrix, userRatingMatrix):
    '''

    :param itemSimilarityMatrix: 物品共现矩阵，DataFrame类型
    :param userRatingMatrix: 用户评分矩阵，DataFrame类型,某一个指定的用户的评分矩阵
    :return: 用户对对应的物品的兴趣值 得到的类型为DataFrame类型，
    '''
    itemNameList = itemSimilarityMatrix.columns
    # print("itemNameList:",itemNameList)
    # 对列名进行重排序，按照物品共现矩阵的列名排列
    singleUserItems = userRatingMatrix[itemNameList]
    # print(singleUserItems)
    print("singleUserItems.values:", singleUserItems.values)
    notViewedMovies = []
    # 过滤掉用户看过的电影
    for col in singleUserItems.columns:
        if singleUserItems[col].all() == 0:
            notViewedMovies.append(col)
    # print(notViewedMovies)
    itemSimilarityMatrix = np.mat(itemSimilarityMatrix.values) #N *N
    # print("itemSimilarityMatrix :", itemSimilarityMatrix)
    singleUserItems = np.mat(singleUserItems.values).T # N * 1
    print("singleUserItems shape:", singleUserItems.shape)
    itemsSimilarScore = itemSimilarityMatrix * singleUserItems # mat生成数组，（*）和np.dot()相同，点乘只能用np.multiply()
    print("itemsSimilarScore:", itemsSimilarScore)
    result = pd.DataFrame(itemsSimilarScore, index=itemNameList, columns=['ratings'])
    result = result.sort_values(by='ratings', ascending=False)
    return result[result.index.isin(notViewedMovies)]

userItemMatrix = getUserItemMatrix(df, "UserID", "MovieID", "Rating")
print(userItemMatrix)
itemNameList = userItemMatrix.columns
print(itemNameList)
similarityMatrix = createItemSimilarityMatrix(userItemMatrix, itemNameList)
print(similarityMatrix)
userResult = getItemCF(similarityMatrix, userItemMatrix.sample(1, axis=0, random_state=1), "MovieID")
print(userResult)
