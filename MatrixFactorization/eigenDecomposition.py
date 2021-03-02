# 特征值分解
#  关于特征值与特征向量的理解见 https://www.zhihu.com/question/21874816/answer/181864044
# 针对方阵，不适合用于分解用户-物品矩阵
import numpy as np
from PIL import Image
def loadDataSet():
    image = Image.open("../data/image/lena.png") # 彩色512*512像素大小
    return np.array(image.convert('L')) # 灰度化

dataSet = loadDataSet()
print(dataSet)
eigVec, eigMat = np.linalg.eig(dataSet) # 特征值组成的向量，特征向量组成的矩阵
print(eigVec)
