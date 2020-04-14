from nlp_pipelines.sweep import sweep


# df = clean_df('/home/dimmu/nlp-pipelines/taxonomyV0.2.csv.gz')
# urls, texts, url2target, uniq_urls, uniq_targets = preprocess_data(mini_df)




vec_params = [
        {'n_range':[5], 'num_buckets':10000}, 
        {'n_range':[4], 'num_buckets':10000}, 
        {'n_range':[4,6], 'num_buckets':10000},
        {'n_range':[5], 'num_buckets':20000}, 
        {'n_range':[4], 'num_buckets':20000}, 
        {'n_range':[4,6], 'num_buckets':20000}
         
]

classifier_params = [0.1,1,5]

sweep('/home/dimmu/testing_df.csv.gz', vec_params, classifier_params, './test_sweep')