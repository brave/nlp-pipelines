from binascii import crc32
import numpy as np 
from scipy import sparse


def ngram_hash(text, n, num_buckets=10000):
    ''' Hash a string of text to a series of integer values
        based on its substrings modulo a number of 'buckets' '''
    rtn = []
    text_bytes = bytes(text, 'utf-8')
    for i in range( len(text_bytes) -n +1):
        fragment = text_bytes[i:(i+n)]
        rtn.append( crc32(fragment) % num_buckets )
    return rtn 

def count_hashed_ngrams(text, n_range=[1,2,3,4,5,6], num_buckets=10000):
    ''' Return a map counting the frequency of hashed ngrams in a string of text 
    '''
    rtn = {}
    for n in n_range:
        tmp = ngram_hash(text, n, num_buckets=num_buckets)
        for t in tmp:
            if t in rtn:
                rtn[t] += 1
            else:
                rtn[t] = 1
    return rtn

def get_dense_hash_count(texts, n_range=[1,2,3,4,5,6], num_buckets=10000):
    rtn = np.zeros( (len(texts), num_buckets) )
    for i, text in enumerate(texts):
        tmp  = count_hashed_ngrams(text,n_range=n_range, num_buckets=num_buckets)
        for idx, count in tmp.items():
            rtn[i,idx] = count
    return sparse.csr_matrix(rtn)

def ngram_hash_count(text, n, num_buckets=10000):
    ''' Hash a string of text to a series of integer values
        based on its substrings modulo a number of 'buckets' '''
    rtn = {}
    for i in range( len(text) -n +1):
        fragment = bytes(text[i:(i+n)], 'utf-8')
        bucket = crc32(fragment) % num_buckets 
        if bucket in rtn:
            rtn[bucket] += 1
        else:
            rtn[bucket] = 1
    idx = sorted(rtn.keys())
    counts = [rtn[i] for i in idx]
    return (idx, counts)

def light_hashed_ngrams(texts, n, num_buckets=10000):
    col_idx = []
    row_idx = []
    cr_counts = []
    for i, text in enumerate(texts): 
        idx, counts = ngram_hash_count(text, n, num_buckets=num_buckets)
        col_idx += idx
        row_idx += [i for element in idx]
        cr_counts += counts
    return sparse.csr_matrix((cr_counts,(row_idx,col_idx)), shape = (len(texts), num_buckets))

def light_hashed_ngram_count(texts, n_range=[5], num_buckets=10000):
    rtn = [] 
    for n in n_range:
        rtn.append(light_hashed_ngrams(texts,n,num_buckets=num_buckets))
    if len(n_range)>1:
        return np.sum(rtn)
    else:
        return rtn[0]