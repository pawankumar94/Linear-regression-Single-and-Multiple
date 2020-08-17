# Implimentation of Linear Regression for Univariate and Multivariate Data 
![Python 3.8.5](https://img.shields.io/badge/Python-3.8.5-blue)
![Docker](https://img.shields.io/badge/Docker%20Engine-19.03.12-blue)
![build](https://img.shields.io/badge/Build-Passing-brightgreen)

## 0. Name
This repository contains the solutions for the assignments intended as a submission. The two tasks are separated using Directories as mentioned below.

    DOCKERIZED_UnivariateLinearRegression : Dockerized implementation of Task 1
    DOCKERIZED_MultivariateLinearRegression : Dockerized implementation of Task 2

***Both the implementations have their own Dockerfiles Please execute them from their respective directories.***

## 1. Description
Each directory (DOCKERIZED_*) provided has these basic contents.
1. A resource directory which is mounted to the container for specifiying the datasets, and to execute matplotlib.pyplot headlessly (*No GUI Adaptation but images piped to this directory*).

       DOCKERIZED_UnivariateLinearRegression\res_univariate\
       DOCKERIZED_MultivariateLinearRegression\res_multivariate\

2. A Dockerfile provided, which deals with the following,
     - Pulling the latest [`python:slim`](https://hub.docker.com/_/python) release  image from dockerhub, slim versions reduce build time and the images generated are relatively small as compared to the [`python:latest`](https://hub.docker.com/_/python) (1.045 GB -> ~100 MB). 
     - Copying the source code to the python image.
     - Installing task specific dependencies.
     - Specifiying a start command via `CMD`.

3. A text file to specify the bare minimum requirements/dependencies for the containerized application to run.
       
       DOCKERIZED_UnivariateLinearRegression\req.txt
       DOCKERIZED_MultivariateLinearRegression\req.txt
      

4.  A python file containing the actual implementation of the Linear Regression with datasets placed in the resource directory, 

		DOCKERIZED_UnivariateLinearRegression\Linear_Regression.py
		DOCKERIZED_MultivariateLinearRegression\LinearRegressionMultipleVariable.py

## 2. Execution
The task can be executed by executing these commands from the task specific directory i.e.,

| **Task** | **Directory**  |
| :------------: | :------------: |
| Uni Variate Linear Regression  | DOCKERIZED_UnivariateLinearRegression\res_univariate\  |
| Multi Variate Linear Regression  |  DOCKERIZED_MultivariateLinearRegression\res_multivariate\  |

## 3. Usage
All the commands, described here are written assuming there docker installed on the source system and the source system runs Microsoft Windows 10 operating system.
1. Building the image

	   docker build .\
2.  Tag the built image

		docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]

3.  Run the selected image and name the container,

		docker run [OPTIONS] IMAGE[:TAG|@DIGEST] [COMMAND] [ARG...]

## 4. Run Options (Necessary)
Options passed to the `docker run` command.

`--name <Container_Name>`

`--mount type=bind,source=<souce_dir>,dir=<target_directory>`

`-e DATA_FILE = resources/<file_name>`

`-e EPOCH_COUNT = <int>`

`-e LEARNING_RATE = <float>`

The option below will be used to test our univariate regressor. Please note that the values in the test dataset are in multiples of 10,000$. so if the desired output is for a population size of 35,000, the program predicts on a value of 3.5. <br>
__(To be used only with Task 1)__

     -e TEST_POPULATION_SIZE = <float> 

The option below will be used to test our multivariate regressor. Please note that this paramaeter expect comma separated values. The format expected will be 
`<Area_in_Sq_foot>,<No_of_Bedrooms>`. The prediction will be cost of the house in $.<br>
__(To be used only with Task 2)__

      -e TEST_FEATURES = <float>,<int>

## 5. Task 1 - UniVariate Regression - Example Case
Assuming, these commands would be executed via the terminal on a Windows OS
 else, please mind the directory separator for Linux(/) and Windows(\\).

Building the docker image
```
docker build .\
```

Tag the built image, `<IMAGE_ID>`, here the tag is  `linearregunivariate:version1.0`
```
docker tag <IMAGE_ID> linearregunivariate:version1.0
```
- Run the generated image with the tag `linearregunivariate:version1.0`. 
- In the command below, the `<source_path>` is the directory where the repo was extracted to. For example if the master was extracted at `C:\Test_Folder\`  i.e, it contains the structure as in the `git@github.com:pawankumar94/Linear-regression-Single-and-Multiple.git:Master` , then `<source_path> = C:\Test_Folder\`
- Target Directory in the docker container is` /resources/` by default and is also used in the `-e DATA_FILE`
- Also make sure the test datasets are placed in the `DOCKERIZED_UnivariateLinearRegression\res_univariate\` and pass the environment variable `-e DATA_FILE`  accordingly. All the contents placed here will be mounted to the container. By default there are files from the assignment included already, this uses `resources/ex1data1.txt`

```
docker run --name LinearRegUnivariate --mount type=bind,source=<source_path>\DOCKERIZED_UnivariateLinearRegression\res_univariate\,target=/resources/ -e DATA_FILE=resources/ex1data1.txt -e EPOCH_COUNT=1000 -e LEARNING_RATE=0.01 -e TEST_POPULATION_SIZE=3.5 linearregunivariate:version1.0
```
- After successful execution of the command, figures from the container will be placed in the `DOCKERIZED_UnivariateLinearRegression\res_univariate\` folder mounted earlier. (Headless)
- Also the output from the python console will be piped to the terminal that was used for docker.

### 5.1 Interpretation of Terminal Output : Task 1
- Optimal Value of m : Denotes the final value of (theta1) i.e. weight in our model <br>
- Optimal Value of c : Denotes the final value of (theta0) i.e. bias in our model <br>
- Beginning Value of Cost : Denotes the value of cost function in the beggining when model starts learning <br>
- Optimal value of Cost : Denotes the final mininmized value of cost function <br>
- Profit :  Denotes the Final profit($) Value for TEST_POPULATION_SIZE <br>

## 6. Task 2 - MultiVariate Regression - Example Case
Assuming, these commands would be executed via the terminal on a Windows OS
 else, please mind the directory separator for Linux(/) and Windows(\\).

Building the docker image
```
docker build .\
```

Tag the built image, `<IMAGE_ID>`, here the tag is  `linearregmultivariate:version1.0`
```
docker tag <IMAGE_ID> linearregmultivariate:version1.0
```
- Run the generated image with the tag `linearregmultivariate:version1.0`. 
- In the command below, the `<source_path>` is the directory where the repo was extracted to. For example if the master was extracted at `C:\Test_Folder\`  i.e, it contains the structure as in the `git@github.com:pawankumar94/Linear-regression-Single-and-Multiple.git:Master` , then `<source_path> = C:\Test_Folder\`
- Target Directory in the docker container is` /resources/` by default and is also used in the `-e DATA_FILE`
- Also make sure the test datasets are placed in the `\DOCKERIZED_MultivariateLinearRegression\res_multivariate\` and pass the environment variable `-e DATA_FILE`  accordingly. All the contents placed here will be mounted to the container. By default there are files from the assignment included already, this uses `resources/ex1data2.txt`

```
docker run --name LinearRegMultivariate --mount type=bind,source=C:\Users\Pawan\Desktop\DOCKERIZED_MultivariateLinearRegression\res_multivariate\,target=/resources/ -e DATA_FILE=resources/ex1data2.txt -e EPOCH_COUNT=1000 -e LEARNING_RATE=0.01 -e TEST_FEATURES=1650.0,3 linearregmultivariate:version1.0
```
- After successful execution of the command, figures from the container will be placed in the `\DOCKERIZED_MultivariateLinearRegression\res_multivariate\` folder mounted earlier. (Headless)
- Also the output from the python console will be piped to the terminal that was used for docker.

### 6.1 Interpretation of Terminal Output : Task 2
- Initial Cost Value : Denotes the value of cost function in the beggining when model starts learning
- Initial Model Parameters : Denotes the initial value of model parameters.
- Final Value of Cost : Denotes the final mininmized value of cost function
- Learnt Model Parameters : Denotes the value of weight and biases after the model converged

## 7. Problem Statement for Univariate :
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.The chain already has trucks in various cities,the file ex1data1.txt consist of Population and the Profits
|   | Population in city 10,000s | Profit in $10,000s |
|:-:|:--------------------------:|:------------------:|
| 0 |           6.1101           |       17.5920      |
| 1 |           5.5277           |       9.1302       |
| 2 |           8.5186           |       13.6620      |

Now we have to use this data to determine which city to expand to next and build model to predict Profit given the population of any city.

## 7.1. Dependancies
* Pandas
* Matplotlib

Installed(via Dockerfile) dependencies using [pip](https://pip.pypa.io/en/stable/)
## 7.2. Initial Data Representation
![alt text](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/readmeGraphics/ex1data1initialdatarep.PNG)

The task of our linear regression model is to learn a hypothesis function(h) which signifies the linear relationship of Profit w.r.t of Population which would be used to predict
the Profit if given the Population of a city , the hypothesis function could be denoted as the simple linear equation of line:
h(x) = theta1( x ) + theta0 ,  where theta0 denotes the bias and theta1 denotes as the weight 
our Hypothesis function is similar to the equation of linear line y = m(x) + c , where m denotes the slope of the line and c denotes the intercept of the line.
The optimization task of our Linear Regression Model is to learn the weights and biases that minimizes the cost function so as to get the best fit value of prediction for our data

*Final Cost Value:*
We have used the MSE as the cost function to determine the difference between our prediction and the actual value from the dataset.
- Beggining Cost Value 32.072733877455654
- Optimal Cost Value 4.516096429262984

## 7.3. Gradient Convergence Plot:
To minimize the cost function value the model needs to learn the best value of the model parameters(theta1,theta0) , initial value of weight(m) and bias(c) is initialized as 0 and these values are updated iteratively to minimize the cost until it becomes constant or minimum.
As below we can see that with our optimized hyperparameter learning rate as *0.01* and epochs as *1000* we could observe a drastic change in our cost Value getting almost constant after *900* epoch

![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/readmeGraphics/gradient%20convergence%20plot%20ex1data.PNG)

*The Final value of model parameters*:
- Optimal value of m(theta1)  1.09872776181806
- Optimal value of c(theta0) -2.957048174396602

## 7.4. *Prediction Value of User input:*
1. Profit on 35, 000 Population :$7041.275642242226 
2. Profit on 70 ,000 Population :$46496.572727228675

## 7.5. Final regression Line
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/readmeGraphics/RegressionLineex1data.PNG)


## 8. Problem Statement for Multivariate Linear Regression:
Suppose you are selling your house and you want to know what a good market price would be.One way to do this is to first collect information on recent houses sold and make a model of housing prices , file ex2data2.txt consist of Population and the Profits
| Size of House(Sq.feet) | Number of Bedrooms | Price of House |
|------------------------|--------------------|----------------|
| 2104                   | 3                  | 399900         |
| 1600                   | 3                  | 329900         |
| 2400                   | 3                  | 369000         |
| 1416                   | 2                  | 232000         |

Now we have to use this data to  and build model to predict Price of the house given the Size and the Number of Bedrooms.

## 8.1. Dependancies
- Pandas
- scikit-learn
- Numpy 
- Matplotlib

Installed(via Dockerfile) dependencies using [pip](https://pip.pypa.io/en/stable/)

## 8.2. Initial Data Representation
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/readmeGraphics/MultivariateDataRep1.PNG)
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/readmeGraphics/MultivariateDataRep2.PNG)

*Relationships,Objective,Cost,Optimization  of our Model for this problem would be same as per described for Univariate. The changes/Updates we would make is in our hypothesis function  and Cost function as we have multiple attributes to deal with , we would learn parameters for each feature to minimize the Cost function*

## 8.3. Feature Scaling 

We could observe from our data that we have skewed distribution of features which would restrict the model to learn weight value for feature with higher scale with a large amount of iterations as compared to features with small scale. Feature Scaling helps in avoiding such problem by normalizing the values of the features within a certain range allowing the gradient to converge much faster.
We have therefore used StandardScalar from **sklearn** library which calculates a standard score for each dependant variable(Area,Bedroom) as follows :

z = (x - u) / s ,{ u : mean , s : Standard deviation, x : sample }

*We would need the value of u and s for each attribute to scale down the input provided to us by the User*

## 8.4. Updated Hypothesis Function(h)

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

*Representation: [theta0 , theta1 , theta2]*
- Initial Values : [0,0,0]
- Final Optimized Value : [340397.96353532 , 108742.65627238 , -5873.22993383]

## 8.5. Gradient Convergence Plot:
The optimized Hyperparameter selected for the model were learning rate as *0.01* and epoch as *1000* 
![alt text ](https://github.com/pawankumar94/Linear-regression-Single-and-Multiple/blob/master/readmeGraphics/gradient%20convergence%20plot%20ex1data2.PNG)

## 8.6. *Prediction Value of User input:*
Price of 1650 Square Foot house with Bedroom : $ 293221.86768361

