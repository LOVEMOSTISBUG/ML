import numpy as np
from tensorflow import keras


x = np.array([[1,18],[0,0],[1,16],[0,0],[1,12],[1,15],[1,12],[0,0],[0,0],[0,0],[0,0],[0,0]])
y = np.array([1,0,1,0,1,1,1,0,0,0,0,0])

#建立三个层级 激活函数sigmoid  relu 
layer_1 = keras.layers.Dense(units=4,activation='sigmoid')
layer_2 = keras.layers.Dense(units=2,activation='sigmoid')
layer_3 = keras.layers.Dense(units=1,activation='sigmoid')

#Input 输入特征是两个 层级目前不懂有什么讲究的
model = keras.Sequential(
    [keras.Input(shape=(2,)),
     layer_1,
     layer_2,
     layer_3
     ]
)
#损失函数就二元交叉 优化器应该是自适应
model.compile(loss = keras.losses.BinaryCrossentropy(),optimizer = keras.optimizers.Adam(learning_rate=0.01)) 
#代入训练数据 100遍
model.fit(x,y,epochs=100)