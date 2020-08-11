import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading data from csv to pandas dataframe
header_list = ["Population", "profit", "error", "m_der"]    #Declaration of the headers for the attributes
dataframe = pd.read_csv(r'ex1data1.txt', sep = ",",names=header_list, header = None)

#initial values of parameters
x = dataframe['Population']
y = dataframe['profit']
m = 0 #slope
c = 0 #intercept
L = 0.01 #learning rate
epoch = 1500
list_m =[]
list_c =[]
for i in range(epoch):
    for ind, val in enumerate(dataframe.Population):
        y_pred = m * val + c                    #Predicted Value of profit
        #print(y_pred)
        y_true = dataframe['profit'][ind]       #True Value of Profit
        error = y_pred - y_true                 #Error Computation
        dataframe['error'][ind] = error         #Appending Values of Error for each instance of Population for commputing Sum of Error
        dataframe['m_der'][ind] = error * val   #Derivation through Mean Squarred Error

    tot_n = (len(dataframe['Population']))           #Total Number of instances

    derivate_m = (L * ((1/tot_n) * (sum(dataframe['m_der']))))
    m = m - derivate_m                               #Updating the Values of slope
    list_m.append(m)
    derivate_c = (L * ((1/tot_n) * (sum(dataframe['error']))))
    c = c - derivate_c                               #Updating the Values of intercept
    list_c.append(c)
#print(list_c)
print(m)
print(c)

predicted_value = m * x + c
predicted_value1 = m * 35000 + c
print(predicted_value1)
plt.title('Profit ratio wrt Population')
plt.scatter(x, y , marker= '+')
#plt.scatter(x,predicted_value)
plt.plot([min(x), max(x)], [min(predicted_value), max(predicted_value)], color='red')  # regression line
plt.show()

