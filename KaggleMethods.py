# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 22:37:38 2019

@author: Andy
"""

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score


def add_missing_columns(df1, df2, default_val=0):
    """
    Adds columns missing from either dataframe so they have the same columns.
    Added columns are given the default value.
    """
    # extract columns
    c1 = df1.columns
    c2 = df2.columns
    
    # record missing columns for data frame 1
    c1_missing = [c2[i] for i in range(len(c2)) if c2[i] not in c1]
    # add them into the data frame 1
    for c in c1_missing:
        df1[c] = default_val
     
    # record missing columns for data frame 2
    c2_missing = [c1[i] for i in range(len(c1)) if c1[i] not in c2]
    # add them into the data frame 2
    for c in c2_missing:
        df2[c] = default_val
    
    # ensure the same ordering of the columns
    new_cols = df1.columns
    df2 = df2[new_cols]
         
    return df1, df2


def ensemble_prediction(trained_clfs, ensemble_clf_inds, X_test):
    """
    Makes prediction of target signal for test data.
    """
    ensemble_clf_inds = ensemble_clf_inds.astype(int)
    n_iter = len(ensemble_clf_inds)
    N = len(X_test)
    pred = np.zeros([n_iter, N])
    for i in range(n_iter):
        clf = trained_clfs[ensemble_clf_inds[i]]
        pred[i,:] = clf.predict(X_test)
        
    # predict by averaging results
    y_pred = np.mean(pred, axis=0)
    
    # ensure values are between 0 and 1
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 1] = 1
    
    return y_pred


from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

def ensemble_selection(trained_clfs, X_valid, y_valid, n_iter, n_warm_start=0,
                       percent_prune=0, tol=1E-4):
    """
    Performs greedy ensemble selection by adding the classifier that improves the AUC 
    over the validation set (X_validate and y_validate) when added to the ensemble until
    reaching for n_iter iterations. Note that models are added with replacement, so the
    best models will be added multiple times. The ensemble predicts the average value of
    its models, which serves as a proxy for the probability.
    
    classifiers : list [] of classifiers that have been trained on separate training data
    X_validate : N x D array of validation data, where N is number of samples and D is number of dimensions
    y_validate : D x 1 array of validation data's true target signal.
    n_iter : number of iterations of ensemble selection to build the ensemble, and thus the number of 
    ensembles added.
    n_warm_start : number of models to add to initial ensemble based on highest
        performance on their own
    percent_prune : percent of models in sample to prune before ensemble selection;
        worst percent_prune% of models will thus not be considered by the model
    tol : tolerance to determine early-stopping. If adding another model does not
        improve the score by at least the tolerance, ensemble selection will stop early.
    
    returns : numpy array of indices of classifiers selected for the ensemble and numpy array of the 
    predictions of the selected models
    """
    assert n_warm_start < n_iter, 'cannot warm start with more models than iterations.'
    assert percent_prune >= 0, 'percent of models to prune must be non-negative.'
    assert percent_prune < 1, 'not all models can be pruned.'
    
    # list of indices of the classifiers selected for the ensemble
    ensemble_clf_inds = np.zeros([n_iter])
    
    # number of samples
    N = len(y_valid)
    
    # initalize array to store predictions of each model in the ensemble
    ensemble_pred = np.zeros([n_iter, N])
    
    # warm start by adding n_warm_start models with best performance
    # data structure to store performance of each classifier
    auc_arr = np.zeros([len(trained_clfs)])
    for i in range(len(trained_clfs)):
        clf = trained_clfs[i]
        auc_arr[i] = roc_auc_score(y_valid, clf.predict(X_valid))
        
    # get indices of models with best performance        
    inds_sort = np.argsort(auc_arr)
    top_n = inds_sort[-n_warm_start:][::-1]
    
    # add best n_warm_start models up front
    for i in range(n_warm_start):
        ensemble_clf_inds[i] = top_n[i]
        ensemble_pred[i] = trained_clfs[top_n[i]].predict(X_valid)
       
    # PRUNING
    n_models = len(inds_sort)
    n_prune = int(percent_prune * n_models)
    inds_keep = inds_sort[n_prune:]
    trained_clfs_pruned = [trained_clfs[i] for i in range(n_models) if i in inds_keep]
    
    # store best score
    max_auc = 0
    # loop through predictions on validation set and greedily add to create ensemble
    for i in range(n_warm_start, n_iter):
        
        # store index of best classifier and its AUC score
        best_clf_index = 0
        diff = 0
        # loop through all classifiers
        for j in range(len(trained_clfs_pruned)):
            clf = trained_clfs_pruned[j]
            
            # add result of current classifier to the ensemble
            ensemble_pred[i] = clf.predict(X_valid)
            
            # average predictions of ensemble for scoring
            y_pred = np.mean(ensemble_pred[:i+1], axis=0)
            
            # score ensemble
            auc = roc_auc_score(y_valid, y_pred)
            print('auc = %f for classifier %i, iteration %i' % (auc, j+1, i+1))
            
            # store model if it outperforms previous models
            if auc > max_auc:
                # update difference from previous maximum
                diff += auc - max_auc
                # store index of high-performing model
                best_clf_index = j
                # update new max score
                max_auc = auc
                
        # record the index of the selected model
        ensemble_clf_inds[i] = best_clf_index
        # add predictions of best model to ensemble
        ensemble_pred[i] = trained_clfs_pruned[best_clf_index].predict(X_valid)
    
        # check if improvement is above tolerance
        if diff <= tol:
            print('improvement is below tolerance. stopping early at iteration %i.' % (i+1))
            ensemble_clf_inds = ensemble_clf_inds[:i+1]
            ensemble_pred = ensemble_pred[:i+1,:]
            break
            
    return ensemble_clf_inds, ensemble_pred


def preprocess_data(filename_train, filename_test, columns=None, prefix=None):
    """
    Preprocesses data located under filename.
    Returns data as input (X).
    If target is True, also returns last column of data as the target signal (y).
    One-hot encodes columns given by "columns" and appends the prefixes given by "prefixes" to their values.
    
    Processing: 1) removes columns with repeated values 2) one-hot encodes categorical columns
    """
    # load data
    df_train = pd.read_csv(filename_train)
    df_test = pd.read_csv(filename_test)
        
    # remove columns with single value in training set from both training and test data
    for col in df_train.columns:
        if len(df_train[col].unique()) == 1:
            df_train.drop(col, inplace=True, axis=1)
            df_test.drop(col, inplace=True, axis=1)

    # extract target signal and remove from data frame
    y_train = df_train.values[:,-1]
    df_train.drop('target', inplace=True, axis=1)
        
    # one-hot encoding - which columns should we one-hot encode???
    if (columns is not None) and (prefix is not None) and (len(columns) == len(prefix)):
        df_train = pd.get_dummies(df_train, columns=columns, prefix=prefix)
        df_test = pd.get_dummies(df_test, columns=columns, prefix=prefix)
        
    df_train, df_test = add_missing_columns(df_train, df_test)
        
    # get numpy array of values  
    X_train = df_train.values
    X_test = df_test.values
    
    return X_train, y_train, X_test


def save_submission_file(savename, predictions):
    """
    Creates .csv file of predictions for submission to Kaggle competition.
    
    savename : name of file to save (should be .csv)
    predictions : N x 1 array of predicted probabilities of voting
    """
    header = 'id,target'
    id_list = np.arange(0,len(predictions)).astype(int)
    submission_data = np.stack((id_list, predictions), axis=-1)

    # save file
    np.savetxt(savename, submission_data, fmt=['%i', '%.10f'], header=header, delimiter=',', comments='')
    

def split_data(X_known, y_known, N_train):
    """
    Splits data into training and validation sets.
    """
    inds = np.arange(0, len(X_known))
    np.random.shuffle(inds)
    inds_train = inds[:N_train]
    inds_valid = inds[N_train:]
    X_train = X_known[inds_train,:]
    y_train = y_known[inds_train]
    X_valid = X_known[inds_valid,:]
    y_valid = y_known[inds_valid]
    
    return X_train, y_train, X_valid, y_valid


def train_classifiers(classifiers, X_train, y_train):
    """
    Train the classifiers in the given list on the given training data.
    
    classifiers : list [] of classifiers
    X_train : N x D array of training data, where N is the number of samples and D is the number of dimensions
    y_train : N x 1 array of target signals corresponding to the training data
    
    returns nothing since it modifies the classifiers in the given list
    """
    # train sample models on cross validated sets from training set
    for i in range(len(classifiers)):
        clf = classifiers[i]
        clf.fit(X_train, y_train)
        print("Fit model %i of %i" % (i+1, len(classifiers)))
