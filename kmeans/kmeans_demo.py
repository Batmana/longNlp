import numpy as np

# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))  # 计算欧氏距离

def kmeans(dataSet, k):
    """

    :param dataSet:
    :param k:
    :param epoch:
    :param glpbalMinDist:
    :return:
    """

    # 获取有多少个点数
    m, n = np.shape(dataSet)
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 随机选择K个质心
    # np.random.uniform 从一个均匀分布上随机采样
    centorids = np.zeros((k, n))
    clusterChange = True
    for i in range(k):
        temp = int(np.random.uniform(0, m))
        centorids[i, :] = dataSet[temp, :]

    while(clusterChange):
        clusterChange = False
        # 计算每个样本到每个质心的欧式距离
        for j in range(m):
            # 计算j 到每个质心到距离,并找出最近的质心
            minDist = 1000000
            minIndex = 0
            for i in range(k):
                distance =distEclud(dataSet[j, :], centorids[i, :])
                if minDist < distance:
                    minDist = distance
                    minIndex = i
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                # 将 j点划入minIndex的簇
                clusterAssment[j, 0] = minIndex
                clusterAssment[j, 1] = minDist**2

        # 重新计算质心
        for i in range(k):
            temp = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0]]  # 获取簇类所有的点
            centorids[i, :] = np.mean(temp, axis=0)  # 对矩阵的行求均值

    return clusterAssment.A[:, 0], centorids

def create_data_set(*cores):
    """生成k-means聚类测试用数据集"""

    ds = list()
    for x0, y0, z0 in cores:
        x = np.random.normal(x0, 0.1 + np.random.random() / 3, z0)
        y = np.random.normal(y0, 0.1 + np.random.random() / 3, z0)
        ds.append(np.stack((x, y), axis=1))

    return np.vstack(ds)

import time
import matplotlib.pyplot as plt

k = 4
ds = create_data_set((0,0,2500), (0,2,2500), (2,0,2500), (2,2,2500))

t0 = time.time()
result, cores = kmeans(ds, k)
t = time.time() - t0

plt.scatter(ds[:,0], ds[:,1], s=1, c=result.astype(np.int))
plt.scatter(cores[:,0], cores[:,1], marker='x', c=np.arange(k))
plt.show()
