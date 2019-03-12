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
   + NOTE: IMPORTANT INFO ABOUT SERIALIZING CLASSIFIERS TO JSON   +
   ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    The c++ implementation of naive bayes used in the browser 
    currently associates a word with its score under the given categories. 
    The serialization to/from json reflects that it's easier to work around 
    that design choice in python than try to refactor c++ code at this point.
    This may change in future versions. 

'''

class Classifier:
    def __init__(self, classifier_type = Classifier_Type.NB, classifier_params = {}):
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params
        self.classifier = build_classifier(classifier_type, classifier_params)
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
        preds = self.classifier.predict(data)
        rtn = []
        for pred in preds: 
            rtn.append(self.label_to_name[pred])
        return rtn
    def to_json(self):
        rtn = {}
        classes = []
        for c in self.classifier.classes_:
            classes.append(self.label_to_name[c])
        rtn['classes'] = classes
        class_weights = {}
        for i, class_name in enumerate(classes):
            class_weights[class_name]=self.classifier.coef_[i,:].tolist()
        rtn['class_weights'] = class_weights
        return json.dumps(rtn)

        # logProbs = {}
        # for dim_counter in range(self.classifier.coef_.shape[1]):
        #     tmp = []
        #     for class_counter in enumerate(classes):
        #         tmp.append(self.classifier.coef_[class_counter, dim_counter])
        #     # indexing by string of the class for compatibility with the c++ version of nb, for now 
        #     logProbs[str(class_counter)]

def load_classifier(json):
    pass
