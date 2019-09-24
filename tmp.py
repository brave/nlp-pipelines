from nlp_pipelines.transformation import To_lower, Hashed_ngrams
from nlp_pipelines.nlp_pipeline import NLP_Model, load_model
import numpy as np
import pandas as pd



tmp = pd.read_csv('japanese_dataset_clean.tsv',sep='\t')
texts = tmp['text']
labels = [i+'-'+j for i, j in zip(list(tmp['category']),list(tmp['subcategory']))]



to_lower = To_lower()
hashed_ngrams = Hashed_ngrams()
pipeline = NLP_Model(language='JA', representation=[to_lower, hashed_ngrams], classifier_type = 'LINEAR')


def minibatch_loader(pipeline, texts,labels, epochs = 5, batch_size=128):
    is_init=False
    uniq_labels = list(set(labels))
    print("uniq_labels = ", uniq_labels)
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        idx = np.random.permutation(len(labels))
        batch_start = 0
        while batch_start<len(labels):
            this_idx = idx[batch_start: (batch_start+batch_size)]
            X = [texts[i] for i in this_idx]
            Y = [labels[i] for i in this_idx]
            pipeline.partial_train(X,Y, uniq_labels)
            batch_start += batch_size
            print('.', end='')
        print('*')



minibatch_loader(pipeline,texts,labels)

