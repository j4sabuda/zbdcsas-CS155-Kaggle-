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


def ensemble_prediction(trained_classifiers, ensemble_clf_inds, X_test):
    """
    Makes prediction of target signal for test data.
    """
    ensemble_clf_inds = ensemble_clf_inds.astype(int)
    n_iter = len(ensemble_clf_inds)
    N = len(X_test)
    pred = np.zeros([n_iter, N])
    for i in range(n_iter):
        clf = trained_classifiers[ensemble_clf_inds[i]]
        pred[i,:] = clf.predict(X_test)
        
    # predict by averaging results
    y_pred = np.mean(pred, axis=0)
    
    return y_pred


def ensemble_selection(trained_classifiers, X_validate, y_validate, n_iter):
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
    
    returns : numpy array of indices of classifiers selected for the ensemble and numpy array of the 
    predictions of the selected models
    """
    # list of indices of the classifiers selected for the ensemble
    ensemble_clf_inds = np.zeros([n_iter])
    
    # number of samples
    N = len(y_validate)
    
    # initalize array to store predictions of each model in the ensemble
    ensemble_pred = np.zeros([n_iter, N])
    
    # loop through predictions on validation set and greedily add to create ensemble
    for i in range(n_iter):
        
        # store index of best classifier and its AUC score
        best_clf_index = 0
        max_auc = 0
        
        # loop through all classifiers
        for j in range(len(trained_classifiers)):
            clf = trained_classifiers[j]
            
            # add result of current classifier to the ensemble
            ensemble_pred[i] = clf.predict(X_validate)
            
            # average predictions of ensemble for scoring
            y_pred = np.mean(ensemble_pred[:i+1], axis=0)
            
            # score ensemble
            auc = roc_auc_score(y_validate, y_pred)
            print('auc = %f for classifier %i, iteration %i' % (auc, j, i))
            
            # store model if it outperforms previous models
            if auc > max_auc:
                best_clf_index = j
                max_auc = auc

        # record the index of the selected model
        ensemble_clf_inds[i] = best_clf_index
        # add predictions of best model to ensemble
        ensemble_pred[i] = trained_classifiers[best_clf_index].predict(X_validate)
    
    return ensemble_clf_inds, ensemble_pred


def preprocess_data(filename_train, filename_test, columns=None, prefixes=None):
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
        
    # one-hot encoding - which columns should we one-hot encode???
    if (columns is not None) and (prefixes is not None) and (len(columns) == len(prefixes)):
        df_train = pd.get_dummies(df_train, columns=columns, prefixes=prefixes)
        df_test = pd.get_dummies(df_train, columns=columns, prefixes=prefixes)
        
    # get numpy array of values  
    X_train = df_train.values[:,:-1]
    y_train = df_train.values[:,-1]
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
