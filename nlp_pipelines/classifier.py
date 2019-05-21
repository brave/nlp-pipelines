from enum import EnumMeta
from sklearn.svm import LinearSVC
import json
import numpy as np
# from . import pipeline_pb2
class Classifier_Type(EnumMeta):
    LINEAR = "LINEAR"# linear svm

def build_classifier(classifier_type, **kwargs):
    if classifier_type == Classifier_Type.LINEAR:
        if 'C' in kwargs:
            model = LinearSVC(C=kwargs['C'])
        else: 
            model = LinearSVC()
        return model 
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
    def __init__(self, classifier_type = Classifier_Type.LINEAR, **kwargs):
        self.classifier_type = classifier_type
        self.classifier_params = kwargs
        self.classifier = build_classifier(classifier_type, **kwargs)
        self.restored_from_file = False
        # self.json_rep=None
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
        if self.restored_from_file:
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
        preds = []
        for c_name, c_weights in self.class_weights.items():
            preds.append(data.dot( np.array(c_weights) ))
        if self.classifier_type == Classifier_Type.NB:
            for i, prior in enumerate(self.class_priors):
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
        rtn['class_weights'] = class_weights
        rtn['class_weights'] = jsonify(class_weights,rounding_precision) 
        if 'class_log_prior_' in self.classifier.__dict__:
            rtn['biases'] = jsonify(self.classifier.class_log_prior_.tolist(), rounding_precision)
        else :
            zero_priors = [0 for i in range(len(classes))]
            rtn['biases'] = jsonify(zero_priors, rounding_precision)
        return rtn
        

        
    # def to_proto(self):
    #     if self.classifier_type == Classifier_Type.NB:
    #         classes = []
    #         for c in self.classifier.classes_:
    #             classes.append(self.label_to_name[c])
    #         vectors = []
    #         for i, _ in  enumerate(classes):
    #             vectors.append(pipeline_pb2.Vector(elements = self.classifier.coef_[i].tolist()))
    #         class_priors = self.classifier.class_log_prior_
    #         nb = pipeline_pb2.Naive_bayes(classes=classes, vectors = vectors, class_priors = class_priors)
    #         return pipeline_pb2.Classifier(nb=nb)


# def classifier_from_proto(classifier_proto):
#     if classifier_proto.HasField('nb'):
#         classifier = Classifier(classifier_type=Classifier_Type.NB)
#         classifier.restored_from_file = True
#         tmp = {}
#         for c_name, c_weights in zip(classifier_proto.nb.classes, classifier_proto.nb.vectors):
#             tmp[c_name] = c_weights.elements
#         classifier.class_weights = tmp
#         classifier.class_priors = classifier_proto.nb.class_priors
#         return classifier
