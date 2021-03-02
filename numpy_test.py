import numpy as np
vector1 = np.array([1, 2, 3, 4])
print(vector1.shape)
linalgNorm2Test = np.linalg.norm(vector1, ord=2)
print(linalgNorm2Test)
print(linalgNorm2Test is 0)

vector2 = np.array([1, 2, 1, 1])
multiplyTest = np.dot(vector1, vector2)
print(multiplyTest)

print(np.equal(vector1,vector2).all())

a = np.array([[1, 2],
              [3, 4]])
b = np.array([[1], [2]]).T
print(a * b)
