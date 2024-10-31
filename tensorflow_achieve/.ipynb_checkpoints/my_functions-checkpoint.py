def print_2d_data(data,title="data"):
    """  data为二维数组  title为字符串  用来打印查看二维数组 从而进行修改 """
    x , y = data.shape
    print("===="*y+"{}".format(title)+"===="*y)
    for i in data :
            for j in i :
                print("|{}|".format(j),end=" ")
            print("\n",end="")
    print("===="*2*y)


if __name__ == "__main__":
    print("我是一个等待调用的包~")
else:
    pass
