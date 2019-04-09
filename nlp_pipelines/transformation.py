from . pipeline_pb2 import Transformation, To_lower, Hash_ngram
from . hash_vectorizer import get_dense_hash_count

HASHED_NGRAMS_DEFAULTS = {'n_range':[1,2,3,4,5,6], 'num_buckets':10000}

def apply_transform(transformation, input_to_transform):
    rtn = []
    if transformation.HasField('to_lower'):
        for inp in input_to_transform:
            rtn.append(inp.lower())
    if transformation.HasField('hash_ngram'):
        rtn = get_dense_hash_count(input_to_transform,
                        n_range=transformation.hash_ngram.hash_sizes,
                        num_buckets=transformation.hash_ngram.num_buckets)
    return rtn