import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing  as sk
import numpy as np

epoch = 400
loss_list = []
learning_rate = 0.01
epoch_list = []
scalar = sk.StandardScaler()  # Initialization of Scaler

# reading data from text file to pandas data frame
dataframe = pd.read_csv(r'ex1data2.txt', sep=',', header= None)
X = dataframe.iloc[:, :-1]  # Dependant Variables
y = dataframe.iloc[:, -1]  # Independant Variables
X = scalar.fit_transform( (X) )   # featureScaling ,helps gradient convergence fast
m, n = X.shape
bias = np.ones((m, 1))  # Initialization of bias
X = np.hstack((bias, X))  # adding bias as feature X0 = 1 for all
m, n = X.shape
theta = np.zeros((1, n))  # weight of hypothesis(Model Parameter)
y = np.reshape(list(y), (len(list(y)), 1))
def cost_computation():
    predicted_value = X.dot(theta.T)  # hypothesis
    error = predicted_value - y
    squared_error = np.square(error)
    sum_squared_error = np.sum(squared_error)
    cost = (1 / (2 * m)) * sum_squared_error  # cost function J(theta)
    return cost
Initial_cost = cost_computation()  # Cost before learning
print('Initial Cost Value', Initial_cost)

# gradient descent algorithm for minimizing the errors(theta)
def gradient_computation(learning_rate, epoch):
    global theta, loss_list, epoch_list
    for i in range(epoch):
        predicted_value = X.dot(theta.T)
        error = predicted_value - y
        squared_error = np.square(error)
        sum_squared_error = np.sum(squared_error)
        cost = (1 / (2 * m)) * sum_squared_error
        gradient = (learning_rate) * ((1 / m) * (np.sum(X * error, axis=0)))  # alpha * partial dervative of theta w.r.t features
        theta = theta - gradient  # weight update
        epoch_list.append(i)
        loss_list.append(cost)
    plt.title('Convergence of Gradient Descent')
    plt.xlabel('NumberofIteration')
    plt.ylabel('CostJ')
    plt.plot(epoch_list, loss_list)  # convergence evaluation
    plt.show()
    
def user_func(user_input):
    user_input = np.reshape(user_input,(-1, 2))
    user_input = scalar.transform(user_input)  # scaling user input with training data mean and standard deviation
    biased = np.ones((len(user_input),1))
    user_input = np.hstack((biased,user_input))  # introducing bias to the user input
    prediction_user_input = user_input.dot(theta.T)  # Prediction of the Price
    return prediction_user_input
gradient_computation(learning_rate, epoch)
print('Learnt Model Parameters',theta)
cost = cost_computation()
print('Optimized  Cost Parameter', cost)
user_input = np.array([1650.0, 3])
profit = user_func(user_input)
print('Cost of the House', profit)