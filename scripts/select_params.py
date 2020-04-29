# small utility file to select the best set of parameters from a given parameter sweep directory
from glob import glob
import numpy as np
from os import path
import json 
import sys

def average_stats(stats_json):
    exclude_keys = set(['accuracy', 'macro avg', 'weighted avg'])
    all_keys = stats_json['results'][0]['rep'].keys()
    all_precisions = {}
    all_recalls = {}
    all_f1s = {}
    for key in all_keys:
        if key not in exclude_keys: 
            all_precisions[key]=[]
            all_recalls[key]=[]
            all_f1s[key]=[]
    for val_loop in stats_json['results']:
        for key in val_loop['rep'].keys():
            if key not in exclude_keys: 
                all_precisions[key].append(val_loop['rep'][key]['precision'])
                all_recalls[key].append(val_loop['rep'][key]['recall'])
                all_f1s[key].append(val_loop['rep'][key]['f1-score'])
        
    precisions = {}
    recalls = {}
    f1s = {}
    for key in all_precisions.keys():
        precisions[key] = np.mean(all_precisions[key])
        recalls[key] = np.mean(all_recalls[key])
        f1s[key] = np.mean(all_f1s[key])
    rtn = {'params': stats_json['params'], 'val_loops': len(stats_json['results']) }
    rtn['precision'] = precisions
    rtn['recall'] = recalls
    rtn['f1s'] = f1s
    rtn['input_path'] = stats_json['input_path']
    rtn['input_column'] = stats_json['input_column']
    rtn['target_column'] = stats_json['target_column']
    return rtn

def weigh_stats(avg_stats_json, key='f1s'):
    stats = avg_stats_json[key]
    mean_stats = np.mean(list(stats.values()))
    return mean_stats

def process_sweep(sweep_dir):
    runs = []
    all_files = glob(path.join(sweep_dir,'*'))
    for file_name in all_files: 
        try: 
            with open(file_name) as f: 
                stats_json = json.loads(f.read())
                stats = average_stats(stats_json)
                runs.append(stats)
        except Exception as e: 
            print('exception: ', e)
    best_rez = 0.0
    best_run = None
    for run in runs: 
        rez = weigh_stats(run)
        if rez > best_rez:
            best_rez = rez
            best_run = run
            print('current best: ', best_rez, ' for ', best_run['params'])
    rtn = {'params': best_run['params']}
    rtn['input_path'] = best_run['input_path']
    rtn['input_column'] = best_run['input_column']
    rtn['target_column'] = best_run['target_column']
    if 'language' not in best_run:
        rtn['language']=''
    else:
        rtn['language']=best_run['language']
    print('final model: ', rtn)
    out_name = sweep_dir+'.selected_params.json'
    with open(out_name,'w') as f: 
        f.write(json.dumps(rtn))
    print('writen to: ', out_name)

if __name__ == "__main__":
    if len(sys.argv)!= 2:
        print('usage: python select_params.py <sweep directory>')
    else:
        process_sweep(sys.argv[1])