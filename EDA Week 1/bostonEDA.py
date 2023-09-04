'''
@author: Manit Mishra
updated 6/17/2023 working with python 3.11
CIS051

The data in this dataset is from the U.S. Census survey around the housing area around Boston MA:
CRIM - per capita crime rate by town, range from 
ZN - proportion of residential land zoned for lots over 25,000 sq.ft., range from 
INDUS - proportion of non-retail business acres per town., range from 
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise), range from 
NOX - nitric oxides concentration (parts per 10 million), range from 
RM - average number of rooms per dwelling, range from 
AGE - proportion of owner-occupied units built prior to 1940, range from 
DIS - weighted distances to five Boston employment centres, range from 
RAD - index of accessibility to radial highways, range from 
TAX - full-value property-tax rate per $10,000, range from 
PTRATIO - pupil-teacher ratio by town, range from
BLACK - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town, range from
LSTAT - % lower status of the population, range from
MEDV - Median value of owner-occupied homes in $1000's, range from 
'''

import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from numpy import set_printoptions
import seaborn as sns
from pandas.plotting import scatter_matrix


#This is the file path of the CSV on my computer
filePath = 'C:/Users/manit/Desktop/Intro to ML Mission College/EDA Week 1/'
filename = 'boston.csv'
data = read_csv(filePath+filename)
names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','RH','TAX','PTRATIO','BLACK','LSTAT','MEDV']

data1 = data.drop(data.columns[0], axis=1) 

# save features as pandas dataframe for stepwise feature selection, This stepwise feature selection did not work, so I just commented out the chunk of code
X1 = data1.drop(data1.columns[13], axis = 1)
Y1 = data1.drop(data1.columns[0:13], axis = 1)

array = data1.values
X = array[:,0:13]
Y = array[:,13]

X1names = X1.columns

peek = data1.head(20)
print(peek)


set_option('display.width', 100)
description = data.describe()
print(description)


plt.figure() # new plot
data1.hist()
plt.show()

#Standardize the Data
scaler1 = StandardScaler().fit(X1)
rescaledX1 = scaler1.transform(X1)

set_printoptions(precision=3)
print(rescaledX1[0:5,:])


dataStandDf = pd.DataFrame(rescaledX1, columns = X1names)


#descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
description = data1.describe()
print(description)


#show descriptive stats after standardization
set_option('display.width', 100)
descriptionStand = dataStandDf.describe()
print(descriptionStand)


# we look at the distribution of data and its descriptive statistics
data1.hist()
plt.show()


# Now plot the histogram after standardization
dataStandDf.hist()
plt.show()


#separate array into input and output components
scaler = Normalizer().fit(X1)
normalizedX = scaler.transform(X1)

#summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5,:])


# you can make a new data frame with the normalized data
dataNormDf = pd.DataFrame(normalizedX, columns = X1names)
dataNormDf['area'] = Y1.values 

# show descriptive stats after standardization
set_option('display.width', 100)
descriptionNorm = dataNormDf.describe()
print(descriptionNorm)

# we look at the distribution of data and its descriptive statistics
dataNormDf.hist()
plt.show()

# correlation heat map, pay attention to correlation between all predicators/features and each predictor and the output
plt.figure() # new plot
corMat = data1.corr(method='pearson')
print(corMat)

# plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()

# scatter plot of all data
plt.figure()
scatter_matrix(data1)
plt.show()