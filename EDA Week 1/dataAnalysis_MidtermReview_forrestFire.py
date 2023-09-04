# -*- coding: utf-8 -*-
"""
@author: jahan
Tested with Python 2.7
updated: 10/19/2020 to work with Python 3.7
CIS051 Midterm Review


For more information, read [Cortez and Morais, 2007].
1. X - x-axis spatial coordinate within the Montesinho park map: 1 to 9
2. Y - y-axis spatial coordinate within the Montesinho park map: 2 to 9
3. month - month of the year: 'jan' to 'dec'
4. day - day of the week: 'mon' to 'sun'
5. FFMC - FFMC index from the FWI system: 18.7 to 96.20
6. DMC - DMC index from the FWI system: 1.1 to 291.3
7. DC - DC index from the FWI system: 7.9 to 860.6
8. ISI - ISI index from the FWI system: 0.0 to 56.10
9. temp - temperature in Celsius degrees: 2.2 to 33.30
10. RH - relative humidity in %: 15.0 to 100
11. wind - wind speed in km/h: 0.40 to 9.40
12. rain - outside rain in mm/m2 : 0.0 to 6.4
13. area - the burned area of the forest (in ha): 0.00 to 1090.84
"""


import pandas as pd
import matplotlib.pyplot as plt
from pandas import set_option
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from numpy import set_printoptions
import seaborn as sns
from pandas.plotting import scatter_matrix

filePath = 'C:/Users/manit/Desktop/Intro to ML Mission College/EDA Week 1/'
filename = 'forestfires.csv'
data1 = read_csv(filePath+filename)
names = ['X','Y','month','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area']
df2 = data1.drop(['month','day'], axis=1)


## save features as pandas dataframe for stepwise feature selection
X1 = data1.drop(['month', 'day','area'], axis = 1)
Y1 = data1.drop(data1.columns[0:12], axis = 1)
X1names = X1.columns

# scaler = StandardScaler().fit(X1)
# rescaledX = scaler.transform(X1)


scaler1 = StandardScaler().fit(X1)
rescaledX1 = scaler1.transform(X1)

# summarize transformed data
set_printoptions(precision=3)
print(rescaledX1[0:5,:])

# you can make a new data frame with the standardized data
dataStandDf = pd.DataFrame(rescaledX1, columns = X1names)
dataStandDf['area'] = Y1.values 

#let's look at the data
peek = df2.head(5)
print(peek)

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
#set_option('display.width', 100)
#set_option('precision', 1)
#description = df2.describe()
#print(description)

# show descriptive stats after standardization
#set_option('display.width', 100)
#set_option('precision', 1)
#descriptionStand = dataStandDf.describe()
#print(descriptionStand)

# we look at the distribution of data and its descriptive statistics
df2.hist()
plt.show()

# Now plot the histogram after standardization
dataStandDf.hist()
plt.show()




# separate array into input and output components
scaler = Normalizer().fit(X1)
normalizedX = scaler.transform(X1)
# summarize transformed data
set_printoptions(precision=3)
print(normalizedX[0:5,:])

# you can make a new data frame with the normalized data
dataNormDf = pd.DataFrame(normalizedX, columns = X1names)
dataNormDf['area'] = Y1.values 

# show descriptive stats after standardization
set_option('display.width', 100)
#set_option('precision', 1)
descriptionNorm = dataNormDf.describe()
print(descriptionNorm)

# we look at the distribution of data and its descriptive statistics
dataNormDf.hist()
plt.show()

# correlation heat map, pay attention to correlation between all predicators/features and each predictor and the output
plt.figure() # new plot
corMat = df2.corr(method='pearson')
print(corMat)
## plot correlation matrix as a heat map
sns.heatmap(corMat, square=True)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title("CORELATION MATTRIX USING HEAT MAP")
plt.show()
#
## scatter plot of all data
plt.figure()
scatter_matrix(data1)
plt.show()

