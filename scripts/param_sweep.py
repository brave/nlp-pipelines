from nlp_pipelines.sweep import sweep


# df = clean_df('/home/dimmu/nlp-pipelines/taxonomyV0.2.csv.gz')
# urls, texts, url2target, uniq_urls, uniq_targets = preprocess_data(mini_df)




vec_params = [
	{'n_range':[4], 'num_buckets':3000} ,
	{'n_range':[4], 'num_buckets':5000} ,
	{'n_range':[4], 'num_buckets':6000} ,
	{'n_range':[4], 'num_buckets':7000} ,
	{'n_range':[4], 'num_buckets':8000} ,
	{'n_range':[4], 'num_buckets':9000} ,
	{'n_range':[4], 'num_buckets':10000} ,
	{'n_range':[4], 'num_buckets':11000} ,
	{'n_range':[4], 'num_buckets':12000} ,
	{'n_range':[5], 'num_buckets':3000} ,
	{'n_range':[5], 'num_buckets':5000} ,
	{'n_range':[5], 'num_buckets':6000} ,
	{'n_range':[5], 'num_buckets':7000} ,
	{'n_range':[5], 'num_buckets':8000} ,
	{'n_range':[5], 'num_buckets':9000} ,
	{'n_range':[5], 'num_buckets':10000} ,
	{'n_range':[5], 'num_buckets':11000} ,
	{'n_range':[5], 'num_buckets':12000} 


]

classifier_params = [0.1,1,5]

#sweep('/home/dimmu/nlp-pipelines/taxonomy_migrations/jp_taxonomyV0.3.csv.gz', vec_params, classifier_params, './sweep0_V3_jp', n_splits=5,  language='JA')
sweep('/home/dimmu/nlp-pipelines/taxonomyV0.3.1.csv.gz', vec_params, classifier_params, './sweep0_V3_en_sizes', n_splits=5, target_column='subcategory', language='EN')
