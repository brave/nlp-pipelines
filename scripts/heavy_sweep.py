from nlp_pipelines.sweep import serial_sweep
from multiprocessing import Pool, TimeoutError






vec_params = [
        {'n_range':[3,5,7], 'num_buckets':10000},
#        {'n_range':[4,6,8], 'num_buckets':10000},
        {'n_range':[3,5,7,9], 'num_buckets':10000}, 
        {'n_range':[4,6,8,10], 'num_buckets':10000} 

]

classifier_params = [0.1,1,5]

#sweep('/home/dimmu/testing_df.csv.gz', vec_params, classifier_params, './test_sweep')
#sweep('/home/dimmu/nlp-pipelines/taxonomyV0.2.csv.gz', vec_params, classifier_params, './sweep0_taxonomyV2_en', n_splits=5)

def run_serial_sweep(sweep_params):
        ( df_path, vec_param, classifier_params, out_path, n_splits )  = sweep_params
        serial_sweep(df_path,vec_param,classifier_params,out_path, n_splits=n_splits)


if __name__ == '__main__':
        df_path = '/home/dimmu/nlp-pipelines/taxonomyV0.3.csv.gz'
        out_path = './sweep0_taxonomyV3_en'
        n_splits = 5
        param_combos = []
        for vec_param in vec_params:
                param_combos.append( (df_path, vec_param, classifier_params, out_path, n_splits) )

#        pool = Pool(processes=2)              # start 3 worker processes
#        pool.map(run_serial_sweep, param_combos)#
     #   pool.close()
      #  pool.terminate()
        for param_combo in param_combos:
                run_serial_sweep(param_combo)
