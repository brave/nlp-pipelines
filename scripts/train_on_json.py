""" Small python utility file to train an example model on 
    a user supplied .csv data file. Mostly demo purposes. 
"""

from nlp_pipelines.transformation import To_lower, Hashed_ngrams, Normalize, clean_texts
from nlp_pipelines.nlp_pipeline import NLP_Model, load_model
import pandas as pd 
import sys,json  
from datetime import datetime

if __name__ == "__main__":
    if len(sys.argv)!= 2:
        print('usage: python train_on_json.py <sweep directory>')
        exit(1)
    with open(sys.argv[1]) as f:
        options_json = json.loads(f.read())
    data_file = options_json['input_path']
    input_column = options_json['input_column']
    target_column = options_json['target_column']
    language = options_json['language']
    output_file = language+'_'+datetime.now().strftime('%Y-%m-%d')+'.json'
    
    print('loading data')
    data_df = pd.read_csv(data_file)
    # recalculate lengths in case of missing entries:
    data_df['length'] = [len(text) for text in data_df[input_column] ]
    data_df=data_df[data_df['length']>150]
    print('Loaded ', len(data_df), ' rows')
    to_lower = To_lower()
    n_range = options_json['params']['vectorizer']['n_range']
    num_buckets = options_json['params']['vectorizer']['num_buckets']
    reg_param = options_json['params']['regularizer']
    hashed_ngrams = Hashed_ngrams(num_buckets=num_buckets, n_range=n_range)
    
    normalize = Normalize()
    model = NLP_Model(language=language, representation=[to_lower, hashed_ngrams, normalize],classifier_type = 'LINEAR', reg_param=reg_param)
    #model = NLP_Model(language=language, representation=[to_lower, hashed_ngrams],classifier_type = 'LINEAR')

    model.classifier.classifier.max_iter=3000 #give you some more room to learn
    texts = data_df[input_column]
    texts = clean_texts(texts)
    labels = data_df[target_column]
    print('training')
    model.train(texts,labels)
    model.save(output_file)
