FROM python:slim

COPY * /
RUN pip install -r req.txt

CMD python Linear_Regression.py DATA_FILE=${DATA_FILE} EPOCH_COUNT=${EPOCH_COUNT} LEARNING_RATE=${LEARNING_RATE} TEST_POPULATION_SIZE=${TEST_POPULATION_SIZE}

# docker tag <image name> linearregunivariate:version1.0
# docker run --name LinearRegUnivariate --mount type=bind,source=C:\Users\Pawan\Desktop\DOCKERIZED_UnivariateLinearRegression\res_univariate\,target=/resources/ -e DATA_FILE=resources/ex1data1.txt -e EPOCH_COUNT=1000 -e LEARNING_RATE=0.01 -e TEST_POPULATION_SIZE=3.5 linearregunivariate:version1.0