import numpy as np
### 定义模型主体部分
### 包括线性回归模型公式、均方损失函数和参数求偏导三部分

def linear_loss(X, y, w, b):
    """
    输入:
    X: 输入变量矩阵
    y: 输出标签变量
    w: 变量参数权重矩阵
    b: 偏置

    输出:
    y_hat: 线性回归模型预测值
    loss: 均方损失
    dw: 权重系数一阶偏导
    db: 偏置一阶偏导
    """

    # 训练样本量
    num_train = X.shape[0]
    # 训练特征数  
    num_feature = X.shape[1]
    # 线性回归预测值
    y_hat = np.dot(X, w) + b
    # 计算预测值与实际标签之间的均方损失
    loss = np.sum((y_hat - y) ** 2) / num_train
    # 基于均方损失对权重系数的一阶梯度
    dw = np.dot(X.T, (y_hat - y)) / num_train
    # 基于均方损失对偏置量的一阶梯度
    db = np.sum((y_hat - y)) / num_train
    return y_hat, loss, dw, db


### 初始化模型参数
def initialize_params(dims):
    """
    输入:
    dims: 训练数据的变量维度
    输出:
    w: 初始化权重系数
    b: 初始化偏置系数
    """
    # 初始化权重系数为零向量
    w = np.zeros((dims, 1))
    # 初始化偏置参数为零
    b = 0
    return w, b


### 定义线性回归模型的训练过程
def linear_train(X, y, learning_rate=0.01, epochs=10000):
    """
    输入:
    X: 输入变量矩阵
    y: 输出标签向量
    learning_rate: 学习率
    epoches: 训练迭代次数
    输出:
    loss_his: 每次迭代的均方损失
    params: 优化后的参数字典
    grads: 优化后的参数梯度字典
    """
    ### 记录训练损失的空列表
    loss_his = []
    ### 初始化整数参数
    w, b = initialize_params(X.shape[1])
    ### 迭代训练
    for i in range(1, epochs):
        # 计算当前迭代的预测值，均方损失和梯度
        y_hat, loss, dw, db = linear_loss(X, y, w, b)
        # 基于梯度下降法的参数更新
        w += -learning_rate * dw
        b += -learning_rate * db

        # 记录当前迭代的损失
        loss_his.append(loss)

        # 每10000次迭代打印当前损失信息
        if i % 10000 == 0:
            print(f"Epoch:{i} loss:{loss}")
        # 将当前迭代步优化后的参数保存到字典中
        params = {
            'w':w,
            'b':b
            }
        # 将当前迭代步的梯度保存到字典中
        grads = {
            'dw':dw,
            'db':db
        }
    return loss_his, params, grads

### 导入数据集

# 导入load_diabietes模块
from sklearn.datasets import load_diabetes
# 导入打乱数据函数
from sklearn.utils import shuffle
# 获取diabetes数据集
diabetes = load_diabetes()
# 获取输入和标签
data, target = diabetes.data, diabetes.target
# 打乱数据集
X, y = shuffle(data, target, random_state = 10)
# 按照8:2划分训练集和测试集
offset = int(X.shape[0] * 0.8)
# 训练集
X_train, y_train = X[:offset], y[:offset]
# 测试集
X_test, y_test = X[offset:], y[offset:]
# 将训练集中的target改为列向量的形式
y_train = y_train.reshape((-1, 1))
# 将测试集中的target改为列向量的形式
y_test = y_test.reshape((-1, 1))
# 打印训练集和测试集的维度
print(f"X_train's shape: {X_train.shape}")
print(f"X_test's shape: {X_test.shape}")
print(f"y_train's shape: {y_train.shape}")
print(f"y_test's shape: {y_test.shape}")


#线性回归模型训练
loss_his, params, grads = linear_train(X_train, y_train, 0.01, 200000)
# 打印训练后得到的模型参数
print(params)
# 定义线性回归模型的预测函数
def predict(X, params):
    """
    输入:
    X: 测试集
    params: 模型训练参数
    输出:
    y_pred: 模型预测结果 
    """
    # 获取模型参数
    w = params['w']
    b = params['b']
    # 预测:
    y_pred= np.dot(X, w) + b
    return y_pred

# 基于测试集的预测
y_pred = predict(X_test, params)
print(y_pred)


### 定义R^2系数函数
def r2_score(y_test, y_pred):
    """
    输入:
    y_test: 测试集标签值
    y_pred: 测试集预测值
    输出:
    r2: R^2系数
    """
    # 测试集标签均值
    y_avg = np.mean(y_test)
    # 总离差平方和
    ss_tot = np.sum((y_test - y_avg) ** 2)
    # 残差平方和
    ss_res = np.sum((y_test - y_pred) ** 2)
    # R^2计算
    r2 = 1 - (ss_res / ss_tot)
    return r2

print(r2_score(y_test, y_pred))