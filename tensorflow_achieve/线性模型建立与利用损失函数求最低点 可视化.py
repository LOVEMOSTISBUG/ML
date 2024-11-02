import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_model_output(x, w, b):
    """输入训练数据x轴 参数w与b 返回预测Y"""
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb


def compute_cost(x, y, w, b): 
    """输入训练XY轴 参数w和b 返回损失函数值"""
    m = x.shape[0] 
    
    cost_sum = 0 
    for i in range(m): 
        f_wb = w * x[i] + b   
        cost = (f_wb - y[i]) ** 2  
        cost_sum = cost_sum + cost  
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost


plt.rcParams['font.family'] = 'STSong'#设置中文字体防止口口
x_train =  np.array([1.0, 2.0,3.0,4.0,5.0])
y_train = np.array([13.2,23.6,45.2,31.5,52.0])


ww = np.arange(-100,100,1)
bb = np.arange(-100,100,1)
W,B = np.meshgrid(ww,bb)#创建两个二维数组
J = compute_cost(x_train,y_train,W, B)

fig = plt.figure()#创建第一个窗口
ax1 = plt.axes(projection = "3d") #创建三维格式对象Axes3D
ax1.set(xlabel='参数W', ylabel='参数B', zlabel='损失函数值J')

ax1.plot_surface(W,B,J,alpha=0.6,cmap="winter")
ax1.contourf(W,B,J,zdir="z",offset=-10,cmap="rainbow") #Z轴投影到WB上
#ax1.contourf(W,B,J,zdir="x",offset=-110,cmap="rainbow")
#ax1.contourf(W,B,J,zdir="y",offset=-110,cmap="rainbow")


plt.figure()#创建第二个窗口
plt.subplot(111)
min_val = np.min(J)#找最低点
min_index = np.unravel_index(np.argmin(J), J.shape)#最低点的索引
min_B,min_W = min_index
tmp_f_wb = compute_model_output(x_train, -100+1*min_W, -100+1*min_B)
plt.plot(x_train, tmp_f_wb, c='b',label='预测值')
plt.scatter(x_train, y_train, marker='x', c='r',label='实际值')
plt.title(f"最小点损失函数数值为：{ min_val},\n其中W,B分别为{-100+1*min_W},{-100+1*min_B}(误差约等于0.1)")
plt.ylabel('Y轴')
plt.xlabel('X轴')


plt.show()