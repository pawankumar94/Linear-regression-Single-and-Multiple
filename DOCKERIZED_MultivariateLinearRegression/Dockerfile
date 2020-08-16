FROM python:slim

COPY * /

RUN pip install -r req.txt

CMD python LinearRegressionMultipleVariable.py DATA_FILE=${DATA_FILE} EPOCH_COUNT=${EPOCH_COUNT} LEARNING_RATE=${LEARNING_RATE} TEST_FEATURES=${TEST_FEATURES}

# docker tag <image name> linearregmultivariate:version1.0
# docker run --name LinearRegMultivariate --mount type=bind,source=C:\Users\Pawan\Desktop\DOCKERIZED_MultivariateLinearRegression\res_multivariate\,target=/resources/ -e DATA_FILE=resources/ex1data2.txt -e EPOCH_COUNT=1000 -e LEARNING_RATE=0.01 -e TEST_FEATURES=1650.0,3 linearregmultivariate:version1.0