# -*- coding: utf-8 -*-
import os
from typing import Union, List

import numpy as np
import pandas as pd
import scipy.stats as st

import itertools
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# This module contains methods to ease more complex analysis of data 
# for data science/statistical cases

def __normalize(serie: pd.Series):

    if serie.std() > 0.:
        return (serie - serie.mean())/serie.std()


def plot_normality(df: pd.DataFrame, names=Union[List,str]):
    """Metodo para plotar visualização de testes de normalidade resumidos
    para colunas de dataframe do pandas

    Args:
        df (pd.DataFrame): [description]
        names ([type], optional): [description]. Defaults to Union[List,str].
    """
    # TODO: incluir validação de nomes 
    # TODO: incluir teste para tipos da lista names

    pd.Series.normalize = __normalize

    colsize = len(names)
    fig, axes = plt.subplots(nrows=colsize, ncols=2, figsize=(20,20));
    
    for i, name in enumerate(names):
        df[name].plot(kind='box', ax=axes[i][0]);
        sm.qqplot(df[name].normalize(), line='45', ax=axes[i][1]);
        
def average_cost(y_test, y_pred, cost=np.eye(2)):
    c = confusion_matrix(y_test,y_pred)
    score = np.sum(c*cost)/np.sum(c)
    return score

def change_threshold(y_true, y_prob, cost, thresholds):
    """Método que cria diversos heatmap considerando vários thresholds
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    plt.figure(figsize=(10,10))

    j = 1
    for i in thresholds:
        i = round(i, 1)
        y_pred = np.where(y_prob >= i, 1, 0)

        plt.subplot(4, 3, j)
        j += 1

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_true,y_pred)
        np.set_printoptions(precision=2)
        
        acc = round(accuracy_score(y_true, y_pred), 2)
        rec = round(recall_score(y_true, y_pred), 2)
        pre = round(precision_score(y_true, y_pred), 2)
        auc = round(roc_auc_score(y_true, y_pred), 2)
        cos = round(average_cost(y_true, y_pred, cost), 2)

        print(f"Threshold: {i} | Roc Auc: {auc:5} | Accuracy: {acc:5} | Recall: {rec:5} | Precision: {pre:5} | Avg cost: {cos}")

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(y_true, y_pred, title=f'Threshold >= {i:.2f}')
    plt.show()


def plot_confusion_matrix(y_test, y_pred, title='Confusion matrix', classes=['Reprov', 'Aprov']):    
    """Método para plotar a matriz de confusão usando matplotlib
    Args:
        y_test (pd.Series): valores verdadeiros do target
        y_pred (pd.Series): valores preditos pelo modelo
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 1.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
