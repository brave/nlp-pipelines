from nlp_pipelines.transformation import To_lower, Hashed_ngrams
from nlp_pipelines.nlp_pipeline import NLP_Model, load_model
import numpy as np
import pandas as pd



tmp = pd.read_csv('japanese_dataset_clean.tsv',sep='\t')
texts = tmp['text']
labels = [i+'-'+j for i, j in zip(list(tmp['category']),list(tmp['subcategory']))]



to_lower = To_lower()
hashed_ngrams = Hashed_ngrams(num_buckets=50000)
pipeline = NLP_Model(language='JA', representation=[to_lower, hashed_ngrams], classifier_type = 'LINEAR')

pipeline.train(texts,labels)
