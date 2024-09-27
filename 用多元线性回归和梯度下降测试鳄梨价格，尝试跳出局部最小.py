import numpy as np
import matplotlib.pyplot as plt
import copy, math


def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b          
        cost = cost + (f_wb_i - y[i])**2      
    cost = cost / (2 * m)                     
    return cost


def compute_gradient(X, y, w, b): 
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        if i% 10 == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing


orig_data = np.loadtxt("data_set/鳄梨价格.csv",dtype=str,skiprows=1,delimiter=",")
X_train =  np.array(np.delete(orig_data,[0,1,2,3,4,5,6,10,11,12],1),dtype=np.float32) #只取小中大包(*_*)
#X_train = X_train * [0.001,0.001,0.001]#原始数据属实太大 乘个系数
X_train[X_train==0]=1
X_train = np.reciprocal(X_train)
X_train[X_train==1]=0
X_train = X_train * [1,10,100]
y_train =  np.array(orig_data[:,1],dtype=np.float32)

print(X_train)

plt.rcParams['font.family'] = 'STSong'#设置中文字体防止口口
plt.figure()


plt.subplot(331)
#initial_w = np.zeros(X_train.shape[1])#一个和特征数量相等的全零矩阵
#initial_b = 0.

initial_w =[1.0,1.0,1.0]
initial_b = 1.0

iterations = 100
alpha = 5.0e-2 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"当迭代{iterations}次 学习率为{alpha}时 prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
plt.plot(np.arange(iterations),J_hist, c='b')
plt.title(f"迭代{iterations}次  学习率为{alpha}")
plt.ylabel('J的值')
plt.xlabel('迭代次数')



plt.show()