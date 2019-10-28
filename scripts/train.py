""" Small python utility file to train an example model on 
    a user supplied .csv data file. Mostly demo purposes. 
"""

from nlp_pipelines.transformation import To_lower, Hashed_ngrams, Normalize
from nlp_pipelines.nlp_pipeline import NLP_Model, load_model
import pandas as pd 
from optparse import OptionParser

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="dataFile",
                  help="file to load training data from, eg --file=<english_language.csv>", 
                  default=None)
    parser.add_option("-i", "--input",
                  dest="inputColumn", default='extracted_text',
                  help="column in csv to use as input, eg: extracted_text")
    parser.add_option("-t", "--target",
                  dest="targetColumn", default='subcategory',
                  help="column in csv to use as target, eg: subcategory")
    parser.add_option("-o", "--output",
                  dest="outputFile", default=None,
                  help="output file to use in name, eg: subcategory")
    parser.add_option("-l", "--language",
                  dest="language", default=None,
                  help="language of the model, eg: EN, JA")    
    (options, args) = parser.parse_args()

    data_file = options.dataFile
    input_column = options.inputColumn
    target_column = options.targetColumn
    output_file = options.outputFile
    language = options.language
    
    if data_file is None:
        print("No input file supplied")
        exit(1)
    if output_file is None: 
        print("No output file specified")
        exit(1)
    
    if language is None:
        print("No language specified")
        exit(1)
    print('loading data')
    data_df = pd.read_csv(data_file)
    if 'length' in data_df.columns:
        data_df=data_df[data_df['length']>150]
    print('Loaded ', len(data_df), ' rows')
    to_lower = To_lower()
    hashed_ngrams = Hashed_ngrams(num_buckets=10000, n_range=[4,5])
    #normalize = Normalize()
    #model = NLP_Model(language=language, representation=[to_lower, hashed_ngrams, normalize],classifier_type = 'LINEAR')
    model = NLP_Model(language=language, representation=[to_lower, hashed_ngrams],classifier_type = 'LINEAR')

    model.classifier.classifier.max_iter=3000 #give you some more room to learn
    texts = data_df[input_column]
    labels = data_df[target_column]
    print('training')
    model.train(texts,labels)
    model.save(output_file)
