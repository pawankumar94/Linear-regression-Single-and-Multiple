# Implimentation of Linear Regression for Univariate and Multivariate Data 
![Python 2.7.15](https://img.shields.io/badge/Python-2.7.15-blue)
## Problem Statement for Univariate :
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.The chain already has trucks in various cities,the file ex1data1.txt consist of Population and the Profits
|   | Population in city 10,000s | Profit in $10,000s |
|:-:|:--------------------------:|:------------------:|
| 0 |           6.1101           |       17.5920      |
| 1 |           5.5277           |       9.1302       |
| 2 |           8.5186           |       13.6620      |

Now we have to use this data to determine which city to expand to next and build model to predict Profit given the population of any city.

## Dependancies
* Pandas
* matplotlib

Install dependencies using [pip](https://pip.pypa.io/en/stable/)
## Initial Data Representation
![alt text](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/ex1data1initialdatarep.PNG)

The task of our linear regression model is to learn a hypothesis function(h) which signifies the linear relationship of Profit w.r.t of Population which would be used to predict
the Profit if given the Population of a city , the hypothesis function could be denoted as the simple linear equation of line:
h(x) = theta1( x ) + theta0 ,  where theta0 denotes the bias and theta1 denotes as the weight 
our Hypothesis function is similar to the equation of linear line y = m(x) + c , where m denotes the slope of the line and c denotes the intercept of the line

*Final Cost Value:*
We have used the MSE as the cost function to determine the difference between our prediction and the actual value from the dataset.
- Beggining Cost Value 32.072733877455654
- Optimal Cost Value 4.516096429262984

## Gradient Convergence Plot:
To minimize the cost function value the model needs to learn the best value of the model parameters(theta1,theta0) , initial value of weight(m) and bias(c) is initialized as 0 and these values are updated iteratively to minimize the cost until it becomes constant or minimum.
As below we can see that with our optimized hyperparameter learning rate as *0.01* and epochs as *1000* we could observe a drastic change in our cost Value getting almost constant after *600* epoch

![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/gradient%20convergence%20plot%20ex1data.PNG)

*The Final value of model parameters*:
- Optimal value of m(theta1)  1.09872776181806
- Optimal value of c(theta0) -2.957048174396602

*Prediction Value of User input:*
1. Profit on 35, 000 Population :$7041.275642242226 
2. Profit on 70 ,000 Population :$46496.572727228675

## Final regression Line
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/RegressionLineex1data.PNG)
