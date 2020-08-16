import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing  as sk
import numpy as np
import sys
import logging
from datetime import datetime

epoch = 500
loss_list = []
learning_rate = 0.01
epoch_list = []
scalar = sk.StandardScaler()  # Initialization of Scaler

# Command Line Argument Parsing
EPOCH_COUNT = -1
DATA_FILE_NAME = -1
LEARNING_RATE = -1
TEST_FEATURES = -1
for each_value in [(str(sys.argv[idx+1])).split("=") for idx in range(len(sys.argv) - 1)]:
    if str(each_value[0]) == 'EPOCH_COUNT':
        EPOCH_COUNT = int(each_value[1])
    if str(each_value[0]) == 'DATA_FILE':
        DATA_FILE_NAME = str(each_value[1])
    if str(each_value[0]) == 'LEARNING_RATE':
        LEARNING_RATE = float(each_value[1])
    if str(each_value[0]) == 'TEST_FEATURES':
        TEST_FEATURES = each_value[1].split(",")
        TEST_FEATURES = [float(each) for each in TEST_FEATURES]

# Assertion
try:
    logging.info("Asserting inputs :")
    assert EPOCH_COUNT != -1
    assert DATA_FILE_NAME != -1
    assert LEARNING_RATE != -1
    assert TEST_FEATURES != -1
    logging.info("Assertion True")
except AssertionError:
    logging.error("Not all inputs are given, Results are not dependable")
    exit(-1)

learning_rate = LEARNING_RATE
epoch = EPOCH_COUNT
# reading data from text file to pandas data frame
dataframe = pd.read_csv(DATA_FILE_NAME, sep=',', header=None)
X = dataframe.iloc[:, :-1]  # Dependant Variables
y = dataframe.iloc[:, -1]  # Independent Variables
X = scalar.fit_transform(X)  # featureScaling ,helps gradient convergence fast
m, n = X.shape
bias = np.ones((m, 1))  # Initialization of bias
X = np.hstack((bias, X))  # adding bias as feature X0 = 1 for theta0 to make all the terms in hypothesis consistent
m, n = X.shape
theta = np.zeros((1, n))  # weight of hypothesis(Model Parameter)
y = np.reshape(list(y), (len(list(y)), 1))
#Initial data Representation:
plt.figure(figsize=(10, 5))
plt.title(' Bedroom vs Price')
plt.xlabel('Bedroom')
plt.ylabel('Price ')
plt.scatter(dataframe.iloc[:, 1:2], y)
# plt.show() # No GUI Adaptation
plt.savefig('./resources/BedroomVsPrice_Multivariate_{0}.png'.format(datetime.utcnow()))
plt.figure(figsize=(10, 5))
plt.title(' Square foot Area vs Price')
plt.xlabel('Area of House')
plt.ylabel('Price ')
plt.scatter(dataframe.iloc[:,:1], y)
# plt.show() # No GUI Adaptation
plt.savefig('./resources/AreaVsPrice_Multivariate_{0}.png'.format(datetime.utcnow()))
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
        gradient = (learning_rate) * ( (1 / m) * (np.sum(X * error, axis=0)))  # alpha * partial dervative of theta w.r.t features
        theta = theta - gradient  # weight update
        epoch_list.append(i)
        loss_list.append(cost)
    print('Initial Value of Cost', loss_list[1])
    print('Final Value of Cost', loss_list[-1])
    plt.figure(figsize=(10, 5))
    plt.title('Convergence of Gradient Descent')
    plt.xlabel('NumberofIteration')
    plt.ylabel('Cost')
    plt.plot(epoch_list, loss_list)  # convergence evaluation
    # plt.show() # No GUI adaption
    plt.savefig('./resources/Covergence_Multivariate_{0}.png'.format(datetime.utcnow()))
def user_func(user_input):
    user_input = np.reshape(user_input, (-1, len(TEST_FEATURES)))
    user_input = scalar.transform(user_input)  # scaling user input with training data mean and standard deviation
    biased = np.ones((len(user_input), 1))
    user_input = np.hstack((biased, user_input))  # introducing bias to the user input
    prediction_user_input = user_input.dot(theta.T)  # Prediction of the Price
    return prediction_user_input
print('Initial Model Parameters', theta)
gradient_computation(learning_rate, epoch)
print('Learnt Model Parameters', theta)
cost = cost_computation()
user_input = np.array(TEST_FEATURES)
profit = user_func(user_input)
print('Cost of the House', profit)
