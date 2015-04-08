
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from random import shuffle
import time

##################################################################################
############# Data Preprocessing #########################
##################################################################################

train = pd.read_csv('sources/train.csv')
test = pd.read_csv('sources/test.csv')

print train
print test

## Obtain histograms and create mappings
cities = sorted(pd.concat([test['City'], train['City']]).unique())
city_types = sorted(pd.concat([test['City Group'], train['City Group']]).unique())
types = sorted(pd.concat([test['Type'], train['Type']]).unique())

# Cities
dic_cities = dict()
i = 0
for x in cities:
    dic_cities[x] = i
    i = i + 1

# City - types
dic_city_types = dict()
i = -1
for x in city_types:
    dic_city_types[x] = i
    i = i + 2

# Types
dic_types = dict()
i = 0
for x in types:
    dic_types[x] = i
    i = i + 1
#dic_types = {'DT': '0 0', 'FC': '1 0', 'IL': '0 1', 'MB': '1 1'}

print "Diccionaries used"
print dic_cities
print dic_city_types
print dic_types

# Get revenues stats
max_revenue = train.revenue.max()
min_revenue = train.revenue.min()
mean_revenue = train.revenue.mean()

## Transform train and test tables to contain the integers
train.insert(1, 'Age', 2015-pd.DatetimeIndex(train['Open Date']).year)
train['Open Date'] = train['Open Date'].str[-4:].astype(int)
train.replace({"City": dic_cities}, inplace=True)
train.replace({"City Group": dic_city_types}, inplace=True)
train.replace({"Type": dic_types}, inplace=True)

# Normalize with gaussian all the elemnts and with max the revenue (there are no negative revenues)
#my_cols = set(train.columns)
#my_cols.remove('revenue')
#my_cols = list(my_cols)
#train[my_cols] = (train[my_cols] - train[my_cols].mean()) / (train[my_cols].max() - train[my_cols].min())
#train.revenue = train.revenue / max_revenue
#train = train / train_orig.max()
train = (train - train.min()) / (train.max() - train.min())

test.insert(1, 'Age', 2015-pd.DatetimeIndex(test['Open Date']).year)
test['Open Date'] = test['Open Date'].str[-4:].astype(int)
test.replace({"City": dic_cities}, inplace=True)
test.replace({"City Group": dic_city_types}, inplace=True)
test.replace({"Type": dic_types}, inplace=True)

test = (test - test.min()) / (test.max() - test.min())
#test = test / train_orig[my_cols].max()

print test


my_cols = set(train.columns)
my_cols.remove('revenue')
my_cols = list(my_cols)
X=train[my_cols].values
Xt=test
y=train['revenue'].values

#randomize the order for cross validation
combined=zip(y,X)
shuffle(combined)
y[:], X[:] = zip(*combined)


#Model Setup
clf = svm.SVR()
scores=[]

ss=KFold(len(y), n_folds=3,shuffle=True)
for trainCV, testCV in ss:
    X_train, X_test, y_train, y_test= X[trainCV], X[testCV], y[trainCV], y[testCV]
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)

    scores.append(mean_squared_error(y_test,y_pred))

#Average RMSE from cross validation
scores=np.array(scores)
print "CV Score:",np.mean(scores**0.5)

#Fit model again on the full training set
clf.fit(X,y)
#Predict test.csv
yp=(clf.predict(Xt) * (max_revenue - min_revenue)) + min_revenue

#Write submission file
sub=pd.read_csv('sources/sampleSubmission.csv')
sub['Prediction']=yp
sub.to_csv("svm-submission" + time.strftime("%Y%m%d-%H%M%S") + ".csv",index=False)