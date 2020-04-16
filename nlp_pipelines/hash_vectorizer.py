from binascii import crc32
import numpy as np 
from scipy import sparse
from collections import Counter

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

def char_ngrams(text_bytes, n_range):
    """Tokenize text_document into a sequence of character n-grams"""
    
    text_len = len(text_bytes)
    ngrams = []

    # bind method outside of loop to reduce overhead
    ngrams_append = ngrams.append

    for n in n_range:
        for i in range(text_len -n +1):
            ngrams_append(text_bytes[i: i + n])
    return ngrams

def light_hashed_ngram_count(texts, n_range=[5], num_buckets=10000):
    col_idx = []
    row_idx = []
    cr_counts = []
    grams = [ char_ngrams(bytes(text,'utf-8'), n_range) for text in texts ]
    col_append = col_idx.append
    row_append = row_idx.append
    cr_append = cr_counts.append 
    for i, this_grams in enumerate(grams):
        counter =  Counter( [ crc32(gram) % num_buckets for gram in this_grams ] )
        for col_el, col_count in list( counter.items() ):
            col_append(col_el)
            row_append(i)
            cr_append(col_count)
    return sparse.csr_matrix((cr_counts,(row_idx,col_idx)), shape = (len(texts), num_buckets), dtype=np.uint16)