import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_sudent_data():
    orig_data = np.loadtxt("data_set/从11个特征预测学生的测试分数.csv",dtype=str,skiprows=1,delimiter=",")
    prepared_data =np.delete(orig_data,[0,2,3,6],1) #学校名 公立私立 教室名  学号 感觉无关数据 删除掉
    prepared_data =np.delete(prepared_data,[-1],1) #实际成绩 删除掉
    #城市和乡村发展程度有一定数值关系 替换为数值
    prepared_data[prepared_data=="Urban"] = 3
    prepared_data[prepared_data=="Suburban"] = 2
    prepared_data[prepared_data=="Rural"] = 1
    #有无资格享受减价/免费午餐
    prepared_data[prepared_data=="Does not qualify"] = 0
    prepared_data[prepared_data=="Qualifies for reduced/free lunch"] = 1
    #先分别提取无序性数据
    class_name = prepared_data[:,1]
    gender= prepared_data[:,3]
    prepared_data = np.delete(prepared_data,[1,3],1)
    oe = OneHotEncoder() #创建一个独热编码器类 因为教室种类和性别并非有序性数据 没有明显的数值差异 所以转更多维度
    #print(class_name)
    #print(class_name.reshape(-1,1))
    oe.fit(class_name.reshape(-1,1))#OneHotEncoder要求输入为二维数组，用reshape重排结构 只有一列的二维数组 
    prepared_class_name = oe.transform(class_name.reshape(-1,1)).toarray()
    #print(prepared_class_name)
    oe.fit(gender.reshape(-1,1))
    prepared_gender = oe.transform(gender.reshape(-1,1)).toarray()
    prepared_data = np.append(prepared_data,prepared_class_name,axis=1)
    prepared_data = np.append(prepared_data,prepared_gender,axis=1)
    X = np.array(prepared_data,dtype=np.float32)
    Y = np.array(orig_data[:,-1],dtype=np.float32)
    return X,Y
