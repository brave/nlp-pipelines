from nlp_pipelines.sweep import sweep


# df = clean_df('/home/dimmu/nlp-pipelines/taxonomyV0.2.csv.gz')
# urls, texts, url2target, uniq_urls, uniq_targets = preprocess_data(mini_df)




vec_params = [
        {'n_range':[4], 'num_buckets':10000}, 
        {'n_range':[6], 'num_buckets':10000}, 
        {'n_range':[8], 'num_buckets':10000},
        {'n_range':[10], 'num_buckets':10000}, 
        {'n_range':[4,8], 'num_buckets':10000},  


]

classifier_params = [0.1,1,5]

sweep('/home/dimmu/nlp-pipelines/taxonomyV0.3.csv.gz', vec_params, classifier_params, './sweep0_V3_en', n_splits=5,  language='EN')
#sweep('/home/dimmu/nlp-pipelines/taxonomyV0.3.csv.gz', vec_params, classifier_params, './sweep0_V3_en', n_splits=5, target_column='predicted_class', language='EN')
