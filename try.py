import numpy as np
from sklearn.preprocessing import OneHotEncoder

def print_data(data,title="data"):
    print("="*20+"{}:".format(title)+"="*20)
    for i in data :
            for j in i :
                print("{}".format(j),end=" ")
            print("\n",end="")
    print("="*20+""+"="*20)



orig_data = np.loadtxt("data_set/从11个特征预测学生的测试分数.csv",dtype=str,skiprows=1,delimiter=",")
print_data(orig_data,"原始数据")
orig_data_1 =np.delete(orig_data,[0,2,3,6],1) #学校名 公立私立 教室名  学号 感觉无关数据 删除掉
              
orig_data_1[orig_data_1=="Urban"] = 3
orig_data_1[orig_data_1=="Suburban"] = 2
orig_data_1[orig_data_1=="Rural"] = 1

print(orig_data_1)


Y =  np.array(orig_data[:,-1],dtype=np.float32) 