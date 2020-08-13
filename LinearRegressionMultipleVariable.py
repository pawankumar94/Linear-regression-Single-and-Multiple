import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
dataframe = pd.read_csv(r'ex1data2.txt', sep=',',header= None)
X = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
m, n = X.shape
theta = np.random.rand(1, n)
#theta = theta.transpose()
print(theta)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
bias = np.ones(1)
count = 1
epoch = 500
loss_list = []
learning_rate = 0.003
X = X.to_numpy()
y = np.reshape(list(y) , (len(list(y)),1))
print(y.shape)
epoch_list = []
for i in range(epoch):
    predicted_value = X.dot(theta.T)+bias
    #predicted_value = predicted_value.to_numpy()
    error = predicted_value - y
    squared_error = np.square(error)
    sum_squarederror = np.sum(squared_error)
    cost = (1 / (2 * m)) * sum_squarederror
    loss_list.append(cost)
    gradient = (learning_rate)*((1/m)*(np.sum(X*error, axis=0)))
    #gradient = gradient.to_numpy()
    theta = theta - gradient
    count+=1
    epoch_list.append(i)


plt.scatter(epoch_list,loss_list)
plt.show()
user_input = np.array([1650, 3])
prediction = user_input.dot(theta.T)+bias
print(prediction)

