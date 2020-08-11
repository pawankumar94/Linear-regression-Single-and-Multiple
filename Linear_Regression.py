import pandas as pd
import matplotlib.pyplot as plt
import sys

header_list = ["Population", "profit", "error", "Squared_Error", "m_der"]  # headers for the attributes
m = 0  # Slope of the line
c = 0  # Intercept of the line
list_m = []
list_c = []
cost_list = []
i_list = []

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
x = dataframe['Population']
y = dataframe['profit']
L = LEARNING_RATE
epoch = EPOCH_COUNT
for i in range(epoch):
    for ind, val in enumerate(dataframe.Population):
        y_pred = m * val + c  # Predicted Value of profit
        y_true = dataframe['profit'][ind]  # True Value of Profit
        error = y_pred - y_true  # Error Computation
        squared_error = error * error
        dataframe['error'][ind] = error
        dataframe['Squared_Error'][ind] = squared_error
        dataframe['m_der'][ind] = error * val  # Partial Derivative through Mean Squared Error
    tot_n = (len(dataframe['Population']))  # Total Number of instances
    sum_squared_error = sum(dataframe['Squared_Error'])
    cost = sum_squared_error/(2*tot_n)
    cost_list.append(cost)  # Computation of Cost Curve
    i_list.append(i)  # Computation of Cost Curve
    derivate_m = (L * ((1/tot_n) * (sum(dataframe['m_der']))))  # Partial Derivative wrt of intercept
    m = m - derivate_m  # Updating the Values of slope
    list_m.append(m)
    derivate_c = (L * ((1/tot_n) * (sum(dataframe['error']))))  # Partial Derivative wrt of intercept
    c = c - derivate_c  # Updating the Values of intercept
    list_c.append(c)
User_profit_prediction = m * TEST_POPULATION_SIZE + c
print('Profit of City in 10,000$', User_profit_prediction)  # Prediction of the Profit wrt of Population
predicted_value = m * x + c  # Predicted value on learned slope and intercept on dataset
print('Optimal value of m ', m)
print('Optimal value of c', c)
# Visualization of Regression Line
plt.title('Profit ratio wrt Population')
plt.xlabel('population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.scatter(x, y, marker='+')
plt.scatter(x, predicted_value)
plt.plot([min(x), max(x)], [min(predicted_value), max(predicted_value)], color='red')  # regression line
plt.show()
# Visualization of Iteration vs Cost
plt.title('Convergence of Gradient descent')
plt.xlabel('iterations')
plt.ylabel('Cost')
plt.plot(i_list, cost_list)
plt.show()