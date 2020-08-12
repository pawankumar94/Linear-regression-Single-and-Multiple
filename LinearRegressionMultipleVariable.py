import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
header_list = ['SizeofHouse(feet)', 'Bedroom', 'Price($1000)']  # headers for the attributes
dataframe = pd.read_csv(r'ex1data2.txt', sep=',', names=header_list)
dataframe[['SizeofHouse(feet)', 'Bedroom']] = scaler.fit_transform(dataframe[['SizeofHouse(feet)', 'Bedroom']])  # Normalization of Data

print(dataframe.head())


