# Implimentation of Linear Regression for Univariate and Multivariate Data 
![Python 2.7.15](https://img.shields.io/badge/Python-2.7.15-blue)
## 1. Problem Statement for Univariate :
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.The chain already has trucks in various cities,the file ex1data1.txt consist of Population and the Profits
|   | Population in city 10,000s | Profit in $10,000s |
|:-:|:--------------------------:|:------------------:|
| 0 |           6.1101           |       17.5920      |
| 1 |           5.5277           |       9.1302       |
| 2 |           8.5186           |       13.6620      |

Now we have to use this data to determine which city to expand to next and build model to predict Profit given the population of any city.

## Dependancies
* Pandas
* Matplotlib

Install dependencies using [pip](https://pip.pypa.io/en/stable/)
## Initial Data Representation
![alt text](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/ex1data1initialdatarep.PNG)

The task of our linear regression model is to learn a hypothesis function(h) which signifies the linear relationship of Profit w.r.t of Population which would be used to predict
the Profit if given the Population of a city , the hypothesis function could be denoted as the simple linear equation of line:
h(x) = theta1( x ) + theta0 ,  where theta0 denotes the bias and theta1 denotes as the weight 
our Hypothesis function is similar to the equation of linear line y = m(x) + c , where m denotes the slope of the line and c denotes the intercept of the line.
The optimization task of our Linear Regression Model is to learn the weights and biases that minimizes the cost function so as to get the best fit value of prediction for our data

*Final Cost Value:*
We have used the MSE as the cost function to determine the difference between our prediction and the actual value from the dataset.
- Beggining Cost Value 32.072733877455654
- Optimal Cost Value 4.516096429262984

## Gradient Convergence Plot:
To minimize the cost function value the model needs to learn the best value of the model parameters(theta1,theta0) , initial value of weight(m) and bias(c) is initialized as 0 and these values are updated iteratively to minimize the cost until it becomes constant or minimum.
As below we can see that with our optimized hyperparameter learning rate as *0.01* and epochs as *1000* we could observe a drastic change in our cost Value getting almost constant after *900* epoch

![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/gradient%20convergence%20plot%20ex1data.PNG)

*The Final value of model parameters*:
- Optimal value of m(theta1)  1.09872776181806
- Optimal value of c(theta0) -2.957048174396602

## *Prediction Value of User input:*
1. Profit on 35, 000 Population :$7041.275642242226 
2. Profit on 70 ,000 Population :$46496.572727228675

## Final regression Line
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/RegressionLineex1data.PNG)


## 2. Problem Statement for Multivariate Linear Regression:
Suppose you are selling your house and you want to know what a good market price would be.One way to do this is to first collect information on recent houses sold and make a model of housing prices , file ex2data2.txt consist of Population and the Profits
| Size of House(Sq.feet) | Number of Bedrooms | Price of House |
|------------------------|--------------------|----------------|
| 2104                   | 3                  | 399900         |
| 1600                   | 3                  | 329900         |
| 2400                   | 3                  | 369000         |
| 1416                   | 2                  | 232000         |

Now we have to use this data to  and build model to predict Price of the house given the Size and the Number of Bedrooms.

## Dependancies
- Pandas
- scikit-learn
- Numpy 
- Matplotlib

Install dependencies using [pip](https://pip.pypa.io/en/stable/)

## Initial Data Representation
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/MultivariateDataRep1.PNG)
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/MultivariateDataRep2.PNG)

*Relationships,Objective,Cost,Optimization  of our Model for this problem would be same as per described for Univariate. The changes/Updates we would make is in our hypothesis function  and Cost function as we have multiple attributes to deal with , we would learn parameters for each feature to minimize the Cost function*

## Feature Scaling 

We could observe from our data that we have skewed distribution of features which would restrict the model to learn weight value for feature with higher scale with a large amount of iterations as compared to features with small scale. Feature Scaling helps in avoiding such problem by normalizing the values of the features within a certain range allowing the gradient to converge much faster.
We have therefore used StandardScalar from **sklearn** library which calculates a standard score for each dependant variable(Area,Bedroom) as follows :

z = (x - u) / s ,{ u : mean , s : Standard deviation, x : sample }

*We would need the value of u and s for each attribute to scale down the input provided to us by the User*

## Updated Hypothesis Function(h)

The earlier hypothesis function described for Univariate Problem was similar to the equation of the linear line with learnable parameters as **Slope** and **Intercept**

*h(x) = theta1(x) + theta0*

The above function works well for data with single feature , we update the equation to deal with multiple features as :

*h(x) = theta0(x0)+ theta1(x1)+theta2(x2)*

*Our Model parameter and Input would be a n+1 dimensional vector*

*We introduced bias = 1 as x0 to provide mathematical convenience and making all the terms in our hypothesis consistent*

*We would update our Cost function  w.r.t the number of feature , so we would have learnable Parameters = Number of features*

*Final Cost Value:*

We have used the **Mean Squarred Error** as the cost function to determine the difference between our prediction and the actual value from the dataset.

- Beggining Cost Value 64297776251.6201
- Optimal Cost Value   2043500888.6116283

*The Final value of model parameters*

### [theta0 , theta1 , theta2]
- Initial Values : [0,0,0]
- Final Optimized Value : [340397.96353532 , 108742.65627238 , -5873.22993383]

## Gradient Convergence Plot:
The optimized Hyperparameter selected for the model were learning rate as *0.01* and epoch as *1000* 
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/gradient%20convergence%20plot%20ex1data2.PNG)

## *Prediction Value of User input:*
Price of 1650 Square Foot house with Bedroom : $ 293221.86768361

