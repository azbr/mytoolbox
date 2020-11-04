import os 
import re
import pandas as pd
from unicodedata import normalize

# This module contains methods to make easier ETL steps on commons data analysis
# cases such as pivoting, cleaning, formatting using numpy arrays and pandas
# Series and DataFrames

def RemoveAcentos(word: str)->str:
    return normalize('NFKD', word).encode('ASCII', 'ignore').decode('ASCII')


def CamelCase(word:str, keeplist={})->str:
    """Converte texto para o formato camelcase
    """
    print(word)
    temp = RemoveAcentos(word)
    print(temp)
    temp = sub('[\(\)]+','', temp)
    print(temp)
    if len(keeplist):
        temp = ''.join([w.title() if w not in keeplist else w for w in temp.split(' ')])
    else:
        temp = ''.join([w.title() for w in temp.split(' ')])
    print(temp)
    return temp
