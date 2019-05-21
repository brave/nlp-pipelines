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
        pass

class Hashed_ngrams:
    def __init__(self, n_range = [1,2,3,4,5,6], num_buckets=[10000]):
        self.n_range = n_range 
        self.num_buckets=num_buckets
    def apply(self, texts):
        return get_dense_hash_count(texts, n_range=[1,2,3,4,5,6], num_buckets=10000)
    def to_json(self):
        pass