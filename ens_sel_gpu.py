# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 17:00:52 2019

@author: Andy
"""

import pandas as pd
import numpy as np

import KaggleMethods as KM

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

# load the data
columns = ['HUFINAL', 'HETENURE', 'HEHOUSUT', 'HETELHHD', 'HETELAVL', 'HEPHONEO', 'HRHTYPE',
          'HUBUS', 'GESTCEN', 'GESTFIPS', 'GTMETSTA', 'PEMARITL', 'PESEX', 'PEAFEVER',
          'PTDTRACE', 'PRDTHSP', 'PRMARSTA', 'PRCITSHP', 'PEMLR', 'PERET1', 'PRCIVLF',
          'PREMPHRS', 'PRWKSTAT', 'PRDTIND1', 'PRDTOCC1', 'PRMJIND1']
prefix = columns
X_known, y_known, X_test = KM.preprocess_data('caltech-cs-155-2019-part-1/train_2008.csv',
                                           'caltech-cs-155-2019-part-1/test_2008.csv',
                                             columns=columns, prefix=prefix)

# number of training samples out of known data (remaining are for validation set)
N_train = 50000
# split known data into training and validation
X_train, y_train, X_valid, y_valid = KM.split_data(X_known, y_known, N_train)

##############################################################################
# collect clfs
clfs = []
############## ADABOOST ####################
# number of estimators
n_estimators = np.arange(10,200, 10)

for i in range(len(n_estimators)):
    clfs += [AdaBoostClassifier(n_estimators=n_estimators[i])]
    
################### DECISION TREES ############################
# list of minimum leaf sizes
min_samples_leaf = np.arange(1, 25, 3)
max_depth = np.arange(1,25,3)

for i in range(len(min_samples_leaf)):
    for j in range(len(max_depth)):
        clfs += [tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=min_samples_leaf[i],
                                                  max_depth=max_depth[j])]
    
#################### Logistic regression models ############
C_arr = np.linspace(0.1, 10, 10)
penalties = ['l1', 'l2']
solver_dict = {'l1':'liblinear', 'l2':'lbfgs'}

for i in range(len(C_arr)):
    for penalty in penalties:
        C = C_arr[i]
        solver = solver_dict[penalty]
        clf = LogisticRegression(solver=solver, penalty=penalty, C=C)
        clfs += [clf]

########### Gaussian Naive Bayes ####################
clfs += [GaussianNB()]

#################### random forest #################
min_samples_split_arr = np.array([2,8,32])
min_samples_leaf_arr = np.array([2,8,32])
n_estimators_arr = np.array([100, 200, 400, 500])

for i in range(len(min_samples_split_arr)):
    for j in range(len(min_samples_leaf_arr)):
        for k in range(len(n_estimators_arr)):
            min_samples_split = min_samples_split_arr[i]
            min_samples_leaf = min_samples_leaf_arr[j]
            n_estimators = n_estimators_arr[k]
            clf = RandomForestClassifier(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                         n_estimators=n_estimators, random_state=1)
            clfs += [clf]

############### REGULARIZED REGRESSION ################
from sklearn.linear_model import Ridge, Lasso

alpha = np.linspace(0.1, 20, 20)
normalize = [True, False]

for i in range(len(alpha)):
    for j in range(len(normalize)):
        clfs += [Ridge(alpha=alpha[i], normalize=normalize[j])]
        clfs += [Lasso(alpha=alpha[i], normalize=normalize[j])]
          
#######################Neural Network##########################
'''
y_train_hot = np.empty([0, 2])
X_train_hot = X_train

for i in range(0, y_train.size):
	y_train_hot = np.vstack((y_train_hot, to_categorical(y_train[i], 2)))
    
y_val_hot = np.empty([0, 2])
X_val_hot = X_valid

for i in range(0, y_valid.size):
	y_val_hot = np.vstack((y_val_hot, to_categorical(y_valid[i], 2)))

#Create Model
model = Sequential()

model.add(Dense(1000, input_shape=(676,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(250))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(125))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(60))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(20))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.4))

model.add(Dense(400))
model.add(Activation('relu'))
model.add(Dropout(0.4))


#Final Layer
model.add(Dense(2))
model.add(Activation('softmax'))

	## Printing a summary of the layers and weights in your model
model.summary()

	#rmsprop and adam optimizers
model.compile(loss='sparse_categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
	
batchSize = 100

#fit = model.fit(X_train_hot, y_train_hot, batch_size=batchSize, nb_epoch=20,
#    verbose=1)

clfs += [model]
'''

###############################################################################
# train
KM.train_classifiers(clfs, X_train, y_train)

# ensemble selection
n_warm_start = 5
n_iter = 200
ensemble_clf_inds, ensemble_pred = KM.ensemble_selection(clfs, X_valid, y_valid, n_iter, 
                                                         n_warm_start=n_warm_start, tol=1E-6)

# predict for test data
y_pred = KM.ensemble_prediction(clfs, ensemble_clf_inds, X_test)


# save file
KM.save_submission_file('submission4.csv', y_pred)
