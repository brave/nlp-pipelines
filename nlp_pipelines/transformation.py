from . hash_vectorizer import get_dense_hash_count

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
    def __init__(self, n_range = [1,2,3,4,5,6], num_buckets=[10000]):
        self.n_range = n_range 
        self.num_buckets=num_buckets
    def apply(self, texts):
        return get_dense_hash_count(texts, n_range=[1,2,3,4,5,6], num_buckets=10000)
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
        rtn = X
        sum_squared = np.sum(X**2, axis = 1)
        for i, s in enumerate(sum_squared):
            rtn[i] = X[i]/s
        return rtn
    def to_json(self):
        return { "transformation_type": "NORMALIZE"}
    


def from_json(transformations_json):
    rtn = []
    for transformation in transformations_json:
        if transformation['transformation_type']=='TO_LOWER':
            rtn.append(To_lower())
        elif transformation['transformation_type']=='HASHED_NGRAMS':
            params = transformation['params']
            rtn.append(Hashed_ngrams(n_range=params['ngrams_range'], num_buckets=params['num_buckets'] ) )
    return rtn
    