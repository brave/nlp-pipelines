from enum import EnumMeta
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier 
import json
import numpy as np

class Classifier_Type(EnumMeta):
    NB = "NB"
    PA = "PA"
    LOGREG = "LOGREG"

def build_classifier(classifier_type, params):
    if classifier_type == Classifier_Type.NB:
        if 'alpha' in params:
            alpha = params['alpha']
        else:
            alpha = 1.0
        if 'fit_prior' in params:
            fit_prior = params['fit_prior']
        else: 
            fit_prior = True
        if 'class_prior' in params:
            class_prior = params['class_prior']
        else: 
            class_prior = None
        return MultinomialNB(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)
    # TODO: Non-default parameters and optimizations for non-NB classifiers
    elif classifier_type == Classifier_Type.PA:
        return PassiveAggressiveClassifier()

    elif classifier_type == Classifier_Type.LOGREG:
        return LogisticRegression()

    else:
        raise ValueError('Unknown classifier type')


'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   +         NOTE:  ABOUT SERIALIZING CLASSIFIERS TO JSON         +
   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    The c++ implementation of naive bayes used in the browser 
    currently associates a word with its score under the given categories. 
    Will be adding a simple inner-product op in the c++ codebase
'''

def jsonify(fp_containing_object,rounding_precision):
    # the following nonsensical line does rounding for json by using full precision float on the dump
    # reading back limited precision float on the read. Apparently they could bother with a parse_float 
    # on only one of the two methods, ergo the roundtrip
    return json.loads(json.dumps(fp_containing_object), parse_float= lambda x:round(float(x), rounding_precision ) )#,  parse_float=lambda x: round(float(x), rounding_precision))))

class Classifier:
    def __init__(self, classifier_type = Classifier_Type.NB, classifier_params = {}):
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params
        self.classifier = build_classifier(classifier_type, classifier_params)
        self.restored_from_json = False
        self.json_rep=None
    def train(self, data, labels):
        uniq_labels = sorted(list(set(labels)))
        self.name_to_labels={ uniq_label:i for i,uniq_label in enumerate(uniq_labels)}
        self.label_to_name={i:uniq_label for i, uniq_label in enumerate(uniq_labels)}
        numeric_labels = []
        for label in labels:
            numeric_labels.append(self.name_to_labels[label])
        numeric_labels = np.array(numeric_labels)
        self.classifier.fit(data, numeric_labels)        
    def predict(self, data):
        if self.restored_from_json:
            return self.predict_from_json(data)
        else:
            return self.predict_from_model(data)
    def predict_from_model(self, data):
        preds = self.classifier.predict(data)
        rtn = []
        for pred in preds: 
            rtn.append(self.label_to_name[pred])
        return rtn
    def predict_from_json(self, data):
        # rtn = np.zeros((len(data), len(self.class_weights)))
        preds = []
        for c_name, c_weights in self.class_weights.items():
            preds.append(data.dot( np.array(c_weights) ))
        if self.classifier_type == Classifier_Type.NB:
            for i, prior in enumerate(self.class_log_prior):
                preds[i] += prior
        preds = np.array(preds)
        max_class = np.argmax(preds,axis=0)
        class_names = list(self.class_weights.keys())
        rtn = []
        for mc in max_class:
            rtn.append(class_names[mc])
        return rtn

    def to_json(self,rounding_precision=4):
        rtn = {}
        rtn['classifier_type']=self.classifier_type
        classes = []
        for c in self.classifier.classes_:
            classes.append(self.label_to_name[c])
        rtn['classes'] = classes
        class_weights = {}
        for i, class_name in enumerate(classes):
            class_weights[class_name]=self.classifier.coef_[i,:].tolist()
        rtn['class_weights'] = jsonify(class_weights,rounding_precision) 
        if 'class_log_prior_' in self.classifier.__dict__:
            rtn['class_log_prior'] = jsonify(self.classifier.class_log_prior_.tolist(), rounding_precision)
        return json.dumps(rtn)

def classifier_from_json(json_string):
    classifier_json = json.loads(json_string)
    classifier = Classifier(classifier_type=classifier_json['classifier_type'])
    classifier.restored_from_json = True
    classifier.class_weights=classifier_json['class_weights']
    if 'class_log_prior' in classifier_json:
        classifier.class_log_prior = classifier_json['class_log_prior']
    return classifier
