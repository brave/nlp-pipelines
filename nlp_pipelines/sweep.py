import pandas as pd 
import numpy as np
from . hash_vectorizer import light_hashed_ngram_count
from . transformation import clean_texts
from sklearn.svm import LinearSVC
# from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize
import json
from hashlib import sha256
import os 
from multiprocessing import cpu_count

def calc_filename(params):
    return sha256(params.encode('utf-8')).hexdigest()

def clean_df(df_path, input_column='extracted_text', min_len=150):
    df = pd.read_csv(df_path)
    lens = [len(text) for text in df[input_column]]
    df['length'] = lens
    df = df[df['length']>=min_len]
    cleaned_texts = clean_texts(list(df[input_column]) )
    df[input_column] = cleaned_texts
    return df

def preprocess_data(df, input_column='extracted_text', target_column='subcategory'):
    """ assuming things are to be indexed by their 'url's:  """
    urls = np.array(df['url'])
    texts = np.array(df[input_column])
    targets = np.array(df[target_column])
    url2target = {url:target for url, target in zip(urls,targets)}
    uniq_urls = np.array(list(set(urls)))
    uniq_targets = np.array([url2target[url] for url in uniq_urls]) 
    return urls, texts, url2target, uniq_urls, uniq_targets

def validate(train_idx, test_idx, urls, uniq_urls, X, url2target, label_encoder,classifier_param):
    train_urls = set(uniq_urls[train_idx]) 
    test_urls = set(uniq_urls[test_idx])
    internal_idx_train = np.where([url in train_urls for url in urls])
    internal_idx_test = np.where([url in test_urls for url in urls])
    X_train = X[internal_idx_train]
    X_test = X[internal_idx_test]
    Y_train = label_encoder.transform([url2target[url] for url in urls[internal_idx_train] ])
    Y_test = label_encoder.transform([url2target[url] for url in urls[internal_idx_test] ])
    model = LinearSVC(C=classifier_param)
    model.fit(X_train,Y_train)
    preds = model.predict(X_test)
    rep = classification_report(label_encoder.inverse_transform(Y_test), 
                                label_encoder.inverse_transform(preds),
                                output_dict=True)
    return dict(rep=rep)

def sweep(df_path, vec_params, classifier_params, out_path, n_splits=10, **kwargs):
    df = clean_df(df_path)
    urls, texts, url2target, uniq_urls, uniq_targets = preprocess_data(df)
    texts = [text.lower() for text in texts]
    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(list(set(uniq_targets))))
    # vectorize
    rtn ={}
    for vec_param in vec_params: 
        n_range = vec_param['n_range']
        num_buckets = vec_param['num_buckets']
        X = light_hashed_ngram_count(texts,n_range=n_range, num_buckets=num_buckets)
        X = normalize(X).astype(np.float32) # check for differences with float16
        n_jobs = min(n_splits, cpu_count())
        for classifier_param in classifier_params:
            skf = StratifiedKFold(n_splits=n_splits)
            out = Parallel(n_jobs=n_jobs, verbose=100, pre_dispatch='1.5*n_jobs')(
                delayed(validate)(train_index, test_index, urls, uniq_urls, X, url2target, label_encoder, classifier_param) for train_index, test_index in skf.split(uniq_urls, uniq_targets))
            param_combo = {'vectorizer':vec_param, 'regularizer':classifier_param}
            to_write = {'params':param_combo}
            to_write['results'] = out
            to_write['input_path'] = df_path
            param_combo = json.dumps(param_combo)
            file_name = calc_filename(param_combo)
            if not os.path.isdir(out_path):
                os.mkdir(out_path)
            with open(os.path.join(out_path, file_name),'w') as f:
                f.write(json.dumps(to_write))
            rtn[param_combo]=out
    return rtn