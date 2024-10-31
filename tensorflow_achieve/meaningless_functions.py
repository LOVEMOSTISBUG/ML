import numpy as np


def cross_validation(data,k,prediction,function):
    """交叉验证 用某个函数计算验证某个学习算法的训练成果和验证集之间的偏差之类的 """
    if data.shape[0]%k == 0 :
        try:
            avg = 0
            for i in range(k):
                X_tain = prediction(np.delete(data,range(i*data.shape[0]/k,(i+1)*data.shape[0]/k)))
                X_test = data[i*data.shape[0]/k,(i+1)*data.shape[0]/k]
                n = function(X_tain,X_test)
                avg = avg + n
            return avg/k
        except:
            print("交叉验证这边出问题了")
    else:
        print("样本数无法与所给折数整除！")
        return 0


def compute_standard_deviation(X,Y):
    """返回两者相减之后的标准差"""
    Z = X - Y
    return np.std(Z)



