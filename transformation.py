from enum import EnumMeta
from hash_vectorizer import get_dense_hash_count
import json

HASHED_NGRAMS_DEFAULTS = {'n_range':[1,2,3,4,5,6], 'num_buckets':10000}

class Transformation_Type(EnumMeta):
    TO_LOWER = "TO_LOWER"
    HASHED_NGRAMS = "HASHED_NGRAMS"

class Transformation:
    def __init__(self, transformation_type=None, params=None):
        self.transformation_type = transformation_type
        self.params = params
    def __repr__(self):
        return self.__dict__.__str__()
    def __str__(self):
        return self.__repr__()
    def apply_transform(self, input_to_transform):
        if self.transformation_type==Transformation_Type.TO_LOWER:
            rtn = []
            for inp in input_to_transform:
                rtn.append(inp.lower())
            return rtn
        if self.transformation_type==Transformation_Type.HASHED_NGRAMS:
            return get_dense_hash_count(input_to_transform,
                        n_range=self.params['n_range'],
                        num_buckets=self.params['num_buckets'])
        raise ValueError('Unknown transformation type')
    def to_json(self):
        return json.dumps(self.__dict__) #hacky but gets the job done

def from_json(json_string):
    tmp = json.loads(json_string)
    if tmp['transformation_type']==Transformation_Type.TO_LOWER:
        return 
    if tmp['transformation_type']==Transformation_Type.HASHED_NGRAMS:
        return 
    raise ValueError('Unknown transformation type')
    