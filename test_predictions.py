""" Utility File to load a produced modej .json file 
    and live test its predictions on a user supplied url """
import numpy as np
from selectolax.parser import HTMLParser
import json 
import gzip
import requests
from nlp_pipelines import hash_vectorizer
from optparse import OptionParser

space_chars = ['\t', '\n', '\r', '\x0b', '\x0c', '\\t', '\\n', '\\r', '\\x0b', '\\x0c']

def remove_junk(content):
    for space_char in space_chars:
        content=content.replace(space_char, ' ')
    return ' '.join(content.split())

def get_page_elements(html):
    tree = HTMLParser(html)
    if tree.body is None:
        return None

    for tag in tree.css('script'):
        tag.decompose()
    for tag in tree.css('style'):
        tag.decompose()

    titles = tree.tags('title')
    if len(titles)>0:
        title = titles[0].text()
    else:
        title = ''
    h_refs = []
    a_tags = tree.tags('a')
    for a_tag in a_tags:
        tag = a_tag.attributes
        if 'href' in tag:
            tmp = {'href': tag['href']}
            if 'title' in tag:
                tmp['title'] = tag['title']
            h_refs.append(tmp)
    meta = {}
    meta_tags = tree.tags('meta')
    for meta_tag in meta_tags:
        attributes = meta_tag.attributes
        if 'property' in attributes:
            if 'content' in attributes:
                property = attributes['property']
                if (property is not None) and (type(property)==str):
                    if 'tag' in property:
                        if attributes['content'] is not None:
                            meta['tags'] = attributes['content'].split(',') 
                    if 'keyword' in property:
                        if attributes['content'] is not None:
                            meta['keywords'] = attributes['content'].split(',')
    tags = ['style', 'script', 'xmp', 'iframe', 'noembed', 'noframes']
    extracted_text = remove_junk(tree.body.text(separator=' '))
    rtn = { 'title':title , 'h_refs':h_refs, 'meta':meta, 'extracted_text':extracted_text}
    return rtn

def load_model(model_path):
    with open(model_path,'r') as f:
        model = json.loads(f.read())
    return model

def extract_text_from_url(url):
    r = requests.get(url)
    tmp = get_page_elements(r.text)
    return tmp['extracted_text']

def convert_to_softmax(rez):
    rtn = {}
    max_rez = np.max(list(rez.values()))
    sum_rez = 0
    for r in rez.values():
        sum_rez += np.exp(r - max_rez)
    for cat, val in rez.items():
        rtn[cat] = np.exp(val - max_rez)/sum_rez
    return rtn

def vectorize_text(text,transformations):
    rtn = text
    for transformation in transformations:
        if transformation['transformation_type'] == 'TO_LOWER':
            print('DOING TO LOWER')
            rtn = rtn.lower()
        elif transformation['transformation_type'] == 'HASHED_NGRAMS':
            print('HASHING NGRAMS, n_grams:',transformation['params']['ngrams_range'],' buckets: ', transformation['params']['num_buckets'] )
            rtn = hash_vectorizer.get_dense_hash_count([rtn], 
                    n_range = transformation['params']['ngrams_range'], 
                    num_buckets = transformation['params']['num_buckets'])
    return rtn

def classify_page(url, model, max_pages = 5):
    extracted_text = extract_text_from_url(url)
    vectorized = vectorize_text(extracted_text, model['transformations'])
    all_rez = {}
    for cat_name, weight in model['classifier']['class_weights'].items():
        rez = vectorized * weight
        all_rez[cat_name] = rez
    softmax_rez = convert_to_softmax(all_rez)
    sorted_rez = sorted(softmax_rez.items(), key = lambda x:x[1])
    selected_rez = sorted_rez[-max_pages:]
    for selected in selected_rez:
        print('category: ', selected[0], ' confidence: ', selected[1])

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="modelFile",
                  help="model file to use, eg --model=<model.json>", 
                  default=None)
    parser.add_option("-u", "--url",
                  dest="evalUrl", default=None,
                  help="url to evaluate eg --url=\"https://www.rakuteneagles.jp\"")
    (options, args) = parser.parse_args()

    model_file = options.modelFile
    if model_file is None:
        print("No modelfile supplied")
        exit(1)
    target_url = options.evalUrl
    if target_url is None: 
        print("No target url supplied")
        exit(1)
    
    try:
        model = load_model(model_file)
    except Exception as e: 
        print('Exception while loading model:', e)
        exit(1)
    
    try: 
        classify_page(target_url, model)
    except Exception as e: 
        print("Exception while trying to classify ", target_url, " : ", e)
        exit(1)


