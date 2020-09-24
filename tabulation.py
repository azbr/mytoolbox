import numpy as np
import pandas as pd
from scipy.stats import kstest

# This module contains methods to make easier ETL steps on commons data analysis
# cases such as pivoting, cleaning, formatting using numpy arrays and pandas
# Series and DataFrames


# Data Analysis Filtering

def getOutliers(serie: pd.Series, dropna=True, dropzeros=True):
    """ Encontra outliers e retorno a o filtro booleano
    """
    lista_quartis = [.25, .5, .75]
    
    if dropna:
        if dropzeros:
            quantis = serie[serie != 0.].dropna().quantile(lista_quartis)
        else:
            quantis = serie.dropna().quantile(lista_quartis)
    
    IQR = quantis[0.75] - quantis[0.25]
    
    return (serie < (quantis[0.25] - 1.5 * IQR)) | (serie > (quantis[0.75] + 1.5 * IQR))

def filterOutliers(serie: pd.Series):
    """Filtra as observações consideradas outliers
    """
    return serie[ getOutliers(serie) ]


def normalize(serie: pd.Series):
    """ Padroniza uma série numérica de dados
    """    
    if serie.std() > 0.:
        return (serie - serie.mean())/serie.std()
    
