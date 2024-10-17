import numpy as np

#skiprows 跳过第几行 delimiter 分隔符号
orig_data = np.loadtxt("data_set/鳄梨价格.csv",dtype=str,skiprows=1,delimiter=",")

price = orig_data[:,1]

#a = np.transpose(orig_data)[2:10]
#print(a)
#a = np.delete(a,[1,2,3],0)
#print(a)
#print(np.transpose(a))


print(orig_data)

price = orig_data[:,1]

print(np.delete(orig_data,[0,1,3,4,5,10,11,12],1)) #删除指定列
a = np.delete(orig_data,[0,1,3,4,5,10,11,12],1)
print(a.shape)

X_train =  np.array(np.delete(orig_data,[0,1,3,4,5,10,11,12],1),dtype=np.float32)
y_train =  np.array(orig_data[:,1],dtype=np.float32)

print(X_train)
print(y_train)