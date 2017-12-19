import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import sklearn.linear_model
import scipy.io

def sigmoid(x):
    """
    实现sigmoid激活函数
    Args:
        x -- scalar或者numpy array类型的输入
    Return:
        s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    """
    实现relu激活函数
    Args:
        x -- scalar或者numpy array类型的输入
    Return:
        s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s

def initialize_parameters(layer_dims):
    """
    初始化模型参数
    Args:
        layer_dims -- 每一网络层的维度信息
    Returns:
        parameters -- 参数信息"W1", "b1", ..., "WL", "bL":
                    W1 -- 权重矩阵，尺寸大小为(layer_dims[l], layer_dims[l-1])
                    b1 -- 偏置向量，尺寸大小为(layer_dims[l], 1)
                    Wl -- 权重矩阵，尺寸大小为(layer_dims[l-1], layer_dims[l])
                    bl -- 偏置向量，尺寸大小为(1, layer_dims[l])
    """
    
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # 网络层数
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == layer_dims[l], layer_dims[l-1])
        assert(parameters['W' + str(l)].shape == layer_dims[l], 1)

        
    return parameters

def forward_propagation(X, parameters):
    """
    实现正向传播过程
    Args:
        X -- 输入数据，尺寸大小为(输入尺寸, 样本数量)
    parameters -- 参数信息 "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- 权重矩阵
                    b1 -- 偏置向量
                    W2 -- 权重矩阵
                    b2 -- 偏置向量
                    W3 -- 权重矩阵
                    b3 -- 偏置向量
    Returns:
        A3 -- 网络最后一层输出的激活函数值，即正向传播的输出
        cache -- 保存的用于计算后向传播过程的信息
    """
        
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache
    
def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    实现采用dropout的正向传播过程
    Args:
        X -- 输入数据，尺寸大小为(输入尺寸, 样本数量)
    parameters -- 参数信息 "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- 权重矩阵
                    b1 -- 偏置向量
                    W2 -- 权重矩阵
                    b2 -- 偏置向量
                    W3 -- 权重矩阵
                    b3 -- 偏置向量
    keep_prob -- dropout操作超参数，取值范围为(0,1]，表示在dropout过程中一个神经元保持激活的概率
    Returns:
        A3 -- 网络最后一层输出的激活函数值，即正向传播的输出
        cache -- 保存的用于计算后向传播过程的信息
    """
    
    np.random.seed(1)
    
    # 获取参数信息
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

      
    D1 = np.random.rand(np.shape(A1)[0], np.shape(A1)[1])       # 步骤1：随机初始化矩阵D1
    D1 = D1 < keep_prob                                         # 步骤2：根据keep_prob大小将D1处理成0-1二值矩阵
    A1 = np.multiply(A1, D1)                                    # 步骤3：暂时抑制A1的部分神经元
    A1 = A1 / keep_prob                                         # 步骤4：调整未被抑制的神经元的输出值

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    D2 = np.random.rand(np.shape(A2)[0], np.shape(A2)[1])       # 步骤1：随机初始化矩阵D2
    D2 = D2 < keep_prob                                         # 步骤2：根据keep_prob大小将D2处理成0-1二值矩阵
    A2 = np.multiply(A2, D2)                                    # 步骤3：暂时抑制A2的部分神经元
    A2 = A2 / keep_prob                                         # 步骤4：调整未被抑制的神经元的输出值

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    
    return A3, cache    

def backward_propagation(X, Y, cache):
    """
    实现反向传播过程
    Args:
        X -- 输入数据，尺寸大小为(输入尺寸, 样本数量)
        Y -- 0-1二值的数据真实标签
        cache -- forward_propagation()输出的cache   
    Returns:
        gradients -- 各种参数、变量对应的梯度
    """
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients

def backward_propagation_with_regularization(X, Y, cache, lambd):
    """
    实现采用L2正则化的反向传播过程
    Args:
        X -- 输入数据，尺寸大小为(输入尺寸, 样本数量)
        Y -- 0-1二值的数据真实标签
        cache -- forward_propagation()输出的cache   
        lambd -- L2正则化超参数
    Returns:
        gradients -- 各种参数、变量对应的梯度
    """
    
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y

    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd / m * W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd / m * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd / m * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients
    
def backward_propagation_with_dropout(X, Y, cache, keep_prob):
    """
    实现采用dropout的反向传播过程
    Args:
        X -- 输入数据，尺寸大小为(输入尺寸, 样本数量)
        Y -- 0-1二值的数据真实标签
        cache -- forward_propagation()输出的cache   
        keep_prob -- dropout操作超参数，取值范围为(0,1]，表示在dropout过程中一个神经元保持激活的概率
    Returns:
        gradients -- 各种参数、变量对应的梯度
    """
    
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)
    dA2 = np.dot(W3.T, dZ3)

    dA2 = np.multiply(dA2, D2)         # 步骤1：利用D2矩阵来暂时抑制与正向传播过程中相同的神经元
    dA2 = dA2 / keep_prob              # 步骤2：调整未被抑制的神经元的输出值

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)
    
    dA1 = np.dot(W2.T, dZ2)

    dA1 = np.multiply(dA1, D1)         # 步骤1：利用D1矩阵来暂时抑制与正向传播过程中相同的神经元
    dA1 = dA1 / keep_prob              # 步骤2：调整未被抑制的神经元的输出值

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    
    return gradients    
    
def update_parameters(parameters, grads, learning_rate):
    """
    利用梯度下降更新参数
    Args:
        parameters -- 参数信息:
                    parameters['W' + str(i)] = Wi
                    parameters['b' + str(i)] = bi
        grads -- 每个参数的梯度:
                    grads['dW' + str(i)] = dWi
                    grads['db' + str(i)] = dbi
        learning_rate -- 学习率
    Returns:
        parameters -- 更新后的参数
    """
    
    n = len(parameters) // 2 # 神经网络的层数

    # 参数更新策略
    for k in range(n):
        parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
        parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
        
    return parameters

def predict(X, y, parameters):
    """
    利用训练好的神经网络模型进行预测
    Args:
        X -- 输入数据
        parameters -- 训练好的模型的参数  
    Returns:
        p -- 预测结果
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # 正向传播
    a3, caches = forward_propagation(X, parameters)
    
    # 根据概率预测标签
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # 打印输出结果
    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def compute_cost(a3, Y):
    """
    计算损失函数cost值 
    Args:
        a3 -- 正向传播的输出，尺寸大小为(输出尺寸, 样本数量)
        Y -- 真实标签向量，尺寸大小和a3相同
    Returns:
        cost - 损失函数cost值
    """
    m = Y.shape[1]
    
    logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1./m * np.nansum(logprobs)
    
    return cost

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    计算采用L2正则化的损失函数cost值
    Args:
        A3 -- 正向传播的输出，尺寸大小为(输出尺寸, 样本数量)
        Y -- 真实标签向量，尺寸大小和a3相同
        parameters -- 模型参数
        lambd -- L2正则化超参数
    Returns:
        cost -- 损失函数cost值
    """
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]
    
    cross_entropy_cost = compute_cost(A3, Y)
    
    L2_regularization_cost = 1. / m * lambd / 2. * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
    
def predict_dec(parameters, X):
    """
    预测分类边界
    Args:
        parameters -- 模型参数
        X -- 输入数据，大小为(m, K)
    Returns:
        predictions -- 预测结果向量(红色: 0 / 蓝色: 1)
    """
    
    # 正向传播并采用0.5作为分类阈值
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3>0.5)
    return predictions

def plot_decision_boundary(model, X, y, figure_name):
    """
    绘制分类边界图像
    Args:
        model -- 训练好的模型
        X -- 输入数据，大小为(m, K)
        y -- 真实标签向量
        figure_name -- 绘制的图像的名称
    Returns:
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.savefig(figure_name)
    
def load_2D_dataset(figure_name):
    """
    读取数据集
    Args:
        figure_name -- 所绘制的数据散点图的名称
    Returns：
        train_X -- 训练数据
        train_Y -- 训练数据的真实标签
        test_X -- 测试数据
        test_Y -- 测试数据的真实标签
    """
    data = scipy.io.loadmat('datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T
    
    #绘制输入数据散点图
    plt.clf()
    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y, s=40, cmap=plt.cm.Spectral);
    plt.savefig(figure_name)
    
    return train_X, train_Y, test_X, test_Y
    