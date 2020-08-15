import pandas as pd
import matplotlib.pyplot as plt
import sys

header_list = ["Population", "profit", "error", "Squared_Error", "m_der"]  # headers for the attributes
m = 0  # slope of the line(weight parameter)
c = 0  # Intercept of the line(bias parameter)
list_m = []
list_c = []
cost_list = []  # used for gradient descent convergence determination
i_list = []  # used for gradient descent convergence determination

# Command Line Argument Parsing
EPOCH_COUNT = -1
DATA_FILE_NAME = -1
LEARNING_RATE = -1
TEST_POPULATION_SIZE = -1
for each_value in [(str(sys.argv[idx+1])).split("=") for idx in range(len(sys.argv) - 1)]:
    if str(each_value[0]) == 'EPOCH_COUNT':
        EPOCH_COUNT = int(each_value[1])
    if str(each_value[0]) == 'DATA_FILE':
        DATA_FILE_NAME = str(each_value[1])
    if str(each_value[0]) == 'LEARNING_RATE':
        LEARNING_RATE = float(each_value[1])
    if str(each_value[0]) == 'TEST_POPULATION_SIZE':
        TEST_POPULATION_SIZE = float(each_value[1])
# reading data from text file to pandas data frame
dataframe = pd.read_csv(DATA_FILE_NAME, sep=",", names=header_list, header=None)
x = dataframe['Population']  # Independant Variable
y = dataframe['profit']  # Dependant Variable on x
L = LEARNING_RATE  # hyperparameter
epoch = EPOCH_COUNT # hyperparameter
for i in range(epoch):
    for ind, val in enumerate(dataframe.Population):
        y_pred = m * val + c  # Predicted Value of profit(hypothesis of Model)
        y_true = dataframe['profit'][ind]  # True Value of Profit
        error = y_pred - y_true  # Error Computation
        squared_error = error * error
        dataframe['error'][ind] = error
        dataframe['Squared_Error'][ind] = squared_error
        dataframe['m_der'][ind] = error * val  # used for Partial derivative w.r.t of slope
    tot_n = (len(dataframe['Population']))  # Total Number of instances
    sum_squared_error = sum(dataframe['Squared_Error'])
    cost = sum_squared_error/(2*tot_n)  # MSE used to penalize large errors
    cost_list.append(cost)
    i_list.append(i)
    derivate_m = (L * ((1/tot_n) * (sum(dataframe['m_der']))))  # Partial Derivative w.r.t of slope
    m = m - derivate_m  # Updating the Values of slope
    list_m.append(m)
    derivate_c = (L * ((1/tot_n) * (sum(dataframe['error']))))  # Partial Derivative w.r.t of intercept
    c = c - derivate_c  # Updating the Values of intercept
    list_c.append(c)
User_profit_prediction = m * TEST_POPULATION_SIZE + c
print('Profit of City  for Population size', (TEST_POPULATION_SIZE*10000) , ':', (User_profit_prediction*10000) )  # Prediction of the Profit wrt of Population
predicted_value = m * x + c  # Predicted value on learned slope and intercept on dataset
print('Optimal value of m :', m)
print('Optimal value of c :', c)
print('Beginning Value of Cost', cost_list[0])
print('optimal Cost value :', cost_list[-1])
#Visualization of Initial Dataset
plt.figure(figsize=(10, 5))
plt.title(' Distribution of data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.scatter(x,y)
plt.show()
# Visualization of Iteration vs Cost
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot()
ax1.set_title('Convergence of Gradient descent')
ax1.set_xlabel('iterations')
ax1.set_ylabel('Cost')
ax1.plot(i_list, cost_list)
plt.show()
# Visualization of Regression Line
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot()
ax.set_title('Profit ratio w.r.t Population')
ax.set_xlabel('Population of City in 10,000s')
ax.set_ylabel('Profit in $10,000s')
ax.scatter(x, y, marker='+', label='Training Data')
#plt.scatter(x, predicted_value)
ax.plot([min(x), max(x)], [min(predicted_value), max(predicted_value)], color='red', label=' Prediction ')  # regression line
ax.legend(loc='best')
plt.show()