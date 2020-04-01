from . hash_vectorizer import get_dense_hash_count, light_hashed_ngram_count
import numpy as np 
from sklearn.preprocessing import normalize as normalize_sk
import re
import string 

class To_lower:
    def __init__(self):
        pass
    def apply(self, texts):
        rtn = []
        for str in texts:
            rtn.append(str.lower())
        return rtn
    def to_json(self):
        return { "transformation_type": "TO_LOWER"}

class Hashed_ngrams:
    def __init__(self, n_range = [5], num_buckets=[10000]):
        self.n_range = n_range 
        self.num_buckets=num_buckets
    def apply(self, texts):
        return light_hashed_ngram_count(texts, n_range=self.n_range, num_buckets=self.num_buckets)
    def to_json(self):
        rtn = { "transformation_type": "HASHED_NGRAMS"}
        rtn['params'] = {}
        rtn['params']['ngrams_range'] = self.n_range
        rtn['params']['num_buckets'] = self.num_buckets
        return rtn

class Normalize: 
    def __init__(self):
        pass 
    def apply(self, X):
        return normalize_sk(X)
    def to_json(self):
        return { "transformation_type": "NORMALIZE"}
    

def clean_texts(texts):
    rx = '[' + re.escape(string.punctuation+'_\t\n\x0b\x0c\r') + ']'
    rxx = rx+'|'+r'[^\w\s]'+'|'+'\S*\d\S*'
    rtn = []
    for text in texts:
        t0 = re.sub('\\\\t\\\\n\\\\v\\\\f\\\\r',' ', text)
        t1 = re.sub(rxx,' ', t0)
        rtn.append(' '.join(t1.split()))
    return rtn


def from_json(transformations_json):
    rtn = []
    for transformation in transformations_json:
        if transformation['transformation_type']=='TO_LOWER':
            rtn.append(To_lower())
        elif transformation['transformation_type']=='HASHED_NGRAMS':
            params = transformation['params']
            rtn.append(Hashed_ngrams(n_range=params['ngrams_range'], num_buckets=params['num_buckets'] ) )
    return rtn
    
