# -*- coding: utf-8 -*-
"""
Linear Regression and Exploratory data analysis
Created on Mon Mar 12 00:53:30 2018
@author: jahan
Tested with Python 2.7
Modified to work with Python 3.6 9/1/18 
Reference for the sample code on step-wise forward model: https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm
"""


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from pandas import read_csv


filename = 'boston.csv'
names = ['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','black','lstat','medv']
data = read_csv(filename)
data1 = data.drop(data.columns[0], axis=1) # when dataframe is read it generates a new index column, this will remove the extra column, check data in variable explorer
# save features as pandas dataframe for stepwise feature selection
X1 = data1.drop(data1.columns[13], axis = 1)
Y1 = data1.drop(data1.columns[0:13], axis = 1)
# separate features and response into two different arrays
array = data1.values
X = array[:,0:13]
Y = array[:,13]
# First perform exploratory data analysis using correlation and scatter plot
# look at the first 20 rows of data
peek = data1.head(20)
print(peek)

# descriptive statistics: mean, max, min, count, 25 percentile, 50 percentile, 75 percentile
set_option('display.width', 100)
#set_option('precision', 1)
description = data.describe()
print(description)

# we look at the distribution of data and its descriptive statistics
plt.figure() # new plot
data1.hist()
plt.show()


# correlation heat map, pay attention to correlation between all predicators/features and each predictor and the output
plt.figure() # new plot
corMat = data1.corr(method='pearson')
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

"""
By observing the preliminary analysis it is obvious that some of the features 
are correlated with each other (colinearity). The ad-hoc appraoch is to keep
only one of the highly correlated features. This approach is only mathematically correct
if the features are identical. Otherwise it is not accurate but good enough for 
a preliminary analysis. The feature selection method described below are two 
approaches that both were discussed in class. Step-wise forward method using
RSS or MSE and  the p-value method. There is a third method and that is called
recursive feature elimination very similar in nature to forward method. Pay attention
that the two methods do not necessarily end up with the same model if the 
parameters happen to be the same number. This has to do with the method of
arriving at the response. The forward method in this case is less accurate since
it is based on p-value only.  

"""
# Determiniation of dominant features , Method one Recursive Model Elimination, 
# very similar idea to foreward selection but done recurssively. This method is gready
# which means it tries one feature at the time
NUM_FEATURES = 6 # this is kind of arbitrary but you should get an idea by observing the scatter plots and correlation.

#14 gave 0.7406
#13 gave 0.7406
#12 gave 0.7406
#11 gave 0.734267
#10 gave 0.72829
#9 gave 0.724344
#8 gave 0.7232588
#7 gave 0.717145
#6 gave 0.71577
#5 gave 0.6336



model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=NUM_FEATURES)
fit = rfe.fit(X, Y)
print("Num Features:", fit.n_features_)
print("Selected Features:", fit.support_)
print("Feature Ranking:", fit.ranking_)
# calculate the score for the selected features
score = rfe.score(X,Y)
print("Model Score with selected features is: ", score)
 

# stepwise forward-backward selection
# need to change the input types as X in this function needs to be a pandas
# dataframe

'''
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(X1, Y1)

print('resulting features:')
print(result)


'''