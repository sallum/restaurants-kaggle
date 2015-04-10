__author__ = 'emulign'

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer, SoftmaxLayer, LinearLayer
from pybrain.tools.validation import CrossValidator, ModuleValidator

import pandas as pd
import time
import random
import numpy as np

##################################################################################
############# Data Preprocessing #########################
##################################################################################

## Read initial data - for submission
train = pd.read_csv('sources/train.csv', index_col=0)
test = pd.read_csv('sources/test.csv', index_col=0)

# For local training use these lines
# local_train = pd.read_csv('sources/train.csv', index_col=0)
# rows = random.sample(local_train.index, 80)
# train = local_train.ix[rows].reset_index()
# train.drop('Id', axis=1, inplace=True)
# test = local_train.drop(rows).reset_index()
# test.drop('Id', axis=1, inplace=True)

#print train
#print test

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
test['Open Date'] = test['Open Date'].str[-4:].astype(int)
test.insert(1, 'Age', 2015 - test['Open Date'])
test.replace({"City": dic_cities}, inplace=True)
test.replace({"City Group": dic_city_types}, inplace=True)
test.replace({"Type": dic_types}, inplace=True)


#train.insert(1, 'Age', 2015-pd.DatetimeIndex(train['Open Date']).year)
train['Open Date'] = train['Open Date'].str[-4:].astype(int)
train.insert(1, 'Age', 2015 - train['Open Date'])
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
my_cols = list(set(train.columns))
my_cols.remove('revenue')

test[my_cols] = (test[my_cols] - train[my_cols].min()) / (train[my_cols].max() - train[my_cols].min())
train.revenue =  np.log(train.revenue)
#train = (train - train.min()) / (train.max() - train.min())
train[my_cols] = (train[my_cols] - train[my_cols].min()) / (train[my_cols].max() - train[my_cols].min())

#test = test / train_orig[my_cols].max()

# print test

##################################################################################
#################### Model Training #########################################
##################################################################################

####################### Multi-layer perceptron #######################
print train
## DataSet
ds = SupervisedDataSet(11, 1)
for i in train.values:
    #print [i[1],i[6],i[34],i[35],i[36]]
    #ds.addSample(tuple(i[1:-1]), i[-1])
    ds.addSample([i[1],i[5],i[6],i[10],i[12],i[15],i[17],i[21],i[25],i[26],i[32]], i[-1])

## Build neural net
net     = buildNetwork(11,
                       30, # number of hidden units
                       1,
                       bias = True,
                       hiddenclass = SigmoidLayer,
                       outclass = LinearLayer
                       )
trainer = BackpropTrainer(net,
                          ds,
                          verbose=True,
                          momentum = 0.1,
                          weightdecay = 0.01)
trainer.trainUntilConvergence( verbose = True, validationProportion = 0.2, maxEpochs = 200, continueEpochs = 10 )
# modval = ModuleValidator()
# for i in range(100):
#     trainer.trainEpochs(1)
#     trainer.trainOnDataset(dataset=ds)
#     cv = CrossValidator( trainer, ds, n_folds=5, valfunc=modval.MSE )
#     print "MSE %f @ %i" %( cv.validate(), i )

####################### SVM #######################


##################################################################################
###################### Prediction ###############################################
##################################################################################

####################### Multi-layer perceptron #######################

## Model Test and prediction
my_cols = list(train[['Age','P1','P2','P6','P8','P11','P13','P17','P21','P22','P28']])
pred = []
for i,t in test.iterrows():
   #print t.Age
   #pred.extend((net.activate(t[1:]) * (max_revenue - min_revenue)) + min_revenue)
   #pred.extend((net.activate(t[1:-1]) * (max_revenue - min_revenue)) + min_revenue)
   pred.extend(np.exp(net.activate(t[my_cols])))
#print pred

####################### SVM #######################



##################################################################################
##################### Writing results ################################
##################################################################################

## Generate Submission file
sub=pd.read_csv('sources/sampleSubmission.csv')
sub['Prediction']=pred
sub.to_csv("submission" + time.strftime("%Y%m%d-%H%M%S") + ".csv",index=False)

## Check error - Local trials
# from sklearn.metrics import mean_squared_error
# from math import sqrt
#
# pred = pd.Series(pred, name = 'Predictions')
# real = (test.revenue * (max_revenue - min_revenue)) + min_revenue
# error = pd.Series(abs(pred - real), name = 'Error')
# results = pd.DataFrame({'Predictions': pred, 'Real': real, 'Error': error})
# print results
# results.to_csv("local-run" + time.strftime("%Y%m%d-%H%M%S") + ".csv", index=False)
#
# print "Error is: "
# print sqrt(mean_squared_error(test.revenue, pred))
