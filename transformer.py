from sklearn.base import BaseEstimator, TransformerMixin
import pandas 
import numpy as np

class Droper(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.list_non_normal = ['6','21','25','33','44','59','65','70']
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        tmp_X = X.copy()
        tmp_X = pandas.DataFrame(data=tmp_X,
                            columns=[str(item) for item in range(0,72)])
        tmp_X.drop(columns=self.list_non_normal, inplace=True)
        col_names = tmp_X.select_dtypes(exclude='object')
        for col in col_names:
            tmp_X[col] = tmp_X[col].astype('float16')
        return tmp_X
    
class Faisser(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        self.dims = X.shape[1]
        self.n_cells = 45
        self.quantizer = faiss.IndexFlatL2(self.dims)
        
        self.model = faiss.IndexIVFFlat(self.quantizer, self.dims, self.n_cells)
        #training
        self.model.train(np.ascontiguousarray(X[:500000, :]))
        self.model.add(np.ascontiguousarray(X))
        
        self.model.nprobe = 12
        
    def predict(self, X):
        vec, idx = self.model.search(np.ascontiguousarray(X), 5)
        return idx