# nlp-pipelines

NLP pipelines & models for in-browser functionality

## installation 

```
pip install .
```

#testing 
to test the example japanese model: 
```
python scripts/test_predictions.py -m models/jp_modelv1.json.gz -u https://www.rakuteneagles.jp/
```
similarly usage for the english model: 

```
python scripts/test_predictions.py -m models/english_hashed.json.gz -u https://brave.com
```

## Background 

This repo contains simple nlp pipelines which can then be packaged for use within the browser
A simple nlp pipeline in this context looks like the following: 

(text)-->(preprocessing)-->(representation)-->(classifier)-->(inferred class)

Briefly, an nlp pipeline defines a series of text transformations to be applied as a preprocessing 
step, eventually producing a numerical representation of the text data. These representations are 
then fed to a classifier which produces a classification rule for the input text data. Some examples 
of text classification pipelines follow:

(japanese text)-->(hashing)-->(naive-bayes)-->result
(english text)-->(lowercase)-->(hashing)-->(linear classifier)-->result

## WHY? 

The major reasons for the existence of this repo are Standardization and reproducibility. 
More precisely:
- standardize on input/output pipelines on nlp along with preprocessing steps
- standardize the way these pipelines are represented, serialized and persisted
- add further metadata to stored models such as language, date produced, processing steps
- make sure the behavior of the python code closely matches the c++ code in the browser
- make it trivial to add new language models both in terms of producing a model and of browser integration

## How does a resulting model look like? 
Currently like the following json object (subject to change to protobuf or other binary serialization):
```
{'time': '2019-03-13 13:24:37.197630',
 'language': 'EN',
 'representation': '{"transforms": ["{"transformation_type": "TO_LOWER", "params": null}", 
                                    "{"transformation_type": "HASHED_NGRAMS", "params": {"n_range": [1, 2, 3, 4, 5], "num_buckets": 500}}"]}',
 'classifier': '{"classifier_type": "NB", 
                "classes": ["ham", "junk", "spam"], 
                "class_weights": {"ham": [-6.779,...., -6.32],
                                  "junk": [...]
                                  "spam":[...]
                    }
                }
}
```

## Testing: 
Just run `pytest`

## TODO:

- Big question about the serialization format currently using json as that was the existing choice for ml models.
  However protobufs or messagepack might result in much shorter models and also substantially simpler to load/serialize/deserialize 
  and cross-function with other languages. 
- Json substantially bigger unless one cuts off a few decimal points <-> accuracy vs stirage concerns
- Add more classifiers and more preprocessing steps

- Downstream book-keeping infrastructure (answer which models work best for which contexts and have those handy)

- Organize into a proper python package and namespace

