import os
from typing import Union, List

import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib.pyplot as plt
import statsmodels.api as sm

# This module contains methods to ease more complex analysis of data 
# for data science/statistical cases


def plot_normality(df: pd.DataFrame, names=Union[List,str]):
    """Metodo para plotar visualização de testes de normalidade resumidos
    para colunas de dataframe do pandas

    Args:
        df (pd.DataFrame): [description]
        names ([type], optional): [description]. Defaults to Union[List,str].
    """
    # TODO: incluir validação de nomes 
    # TODO: incluir teste para tipos da lista names

    colsize = len(names)
    fig, axes = plt.subplots(nrows=colsize, ncols=2, figsize=(20,20));
    
    for i, name in enumerate(names):
        df[name].plot(kind='box', ax=axes[i][0]);
        sm.qqplot(df[name].normalize(), line='45', ax=axes[i][1]);