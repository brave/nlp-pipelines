from binascii import crc32
import numpy as np 


def ngram_hash(text, n, num_buckets=10000):
    ''' Hash a string of text to a series of integer values
        based on its substrings modulo a number of 'buckets' '''
    rtn = []
    for i in range( len(text) -n +1):
        fragment = bytes(text[i:(i+n)], 'utf-8')
        rtn.append( crc32(fragment) % num_buckets )
        # print("fragment: ", fragment, " hash: ", crc32(fragment) % num_buckets)
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
    return rtn
