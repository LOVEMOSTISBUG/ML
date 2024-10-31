import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_sudent_data():
    """"返回已经经过预处理的数据"""
    orig_data = np.loadtxt("data_set/从11个特征预测学生的测试分数.csv",dtype=str,skiprows=1,delimiter=",")
    prepared_data =np.delete(orig_data,[0,2,3,6],1)
    prepared_data =np.delete(prepared_data,[-1],1) 
    prepared_data[prepared_data=="Urban"] = 3
    prepared_data[prepared_data=="Suburban"] = 2
    prepared_data[prepared_data=="Rural"] = 1
    prepared_data[prepared_data=="Does not qualify"] = 0
    prepared_data[prepared_data=="Qualifies for reduced/free lunch"] = 1
    class_name = prepared_data[:,1]
    gender= prepared_data[:,3]
    prepared_data = np.delete(prepared_data,[1,3],1)
    oe = OneHotEncoder()
    oe.fit(class_name.reshape(-1,1)) 
    prepared_class_name = oe.transform(class_name.reshape(-1,1)).toarray()
    oe.fit(gender.reshape(-1,1))
    prepared_gender = oe.transform(gender.reshape(-1,1)).toarray()
    prepared_data = np.append(prepared_data,prepared_class_name,axis=1)
    prepared_data = np.append(prepared_data,prepared_gender,axis=1)
    X = np.array(prepared_data,dtype=np.float32)
    Y = np.array(orig_data[:,-1],dtype=np.float32)
    return X,Y

