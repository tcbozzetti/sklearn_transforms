from sklearn.base import BaseEstimator, TransformerMixin

import numpy
import pandas

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class FillMissingGrade(BaseEstimator, TransformerMixin):
    def fill(self, row):
        if numpy.isnan(row['NOTA_GO']):
            return (row['NOTA_DE'] + row['NOTA_EM'] + row['NOTA_MF']) / 3
        else:
            return row['NOTA_GO']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data['NOTA_GO'] = data.apply(self.fill, axis=1)
        return data

class NormalizeColumn(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        data['INGLES'].fillna('NAO_INFORMADO', inplace = True)
        data['INGLES'] = data['INGLES'].replace([0], 'NAO')
        data['INGLES'] = data['INGLES'].replace([1], 'SIM')
        return pandas.get_dummies(data, columns=['INGLES'])