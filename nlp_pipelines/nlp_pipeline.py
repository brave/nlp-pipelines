from . classifier import Classifier, Classifier_Type
from enum import Enum
from datetime import datetime
import json
#from . pipeline_pb2 import Pipeline

PIPELINE_VERSION = 1

class Language(Enum):
    EN = "EN"
    DE = "DE"
    FR = "FR"
    JA = "JA"

class NLP_Model:
    def __init__(self, language, representation=None, 
                classifier_type=Classifier_Type.LINEAR, version = PIPELINE_VERSION, 
                classifier = None):
        if language not in Language.__members__:
            raise ValueError('Unknown language')
        self.version = version
        self.language = language
        self.representation=representation
        if classifier is None:
            self.classifier = Classifier(classifier_type=classifier_type)
        else:
            self.classifier = classifier
    def apply_transforms(self, data):
        tmp = data
        for transform in self.representation:
            tmp = transform.apply( tmp )
        return tmp
    def train(self, data, labels):
        rep = data
        if self.representation is not None:
            rep = self.apply_transforms(data)
        self.classifier.train(rep,labels)
    def predict(self, data):
        rep = data
        if self.representation is not None:
            rep = self.apply_transforms(data)
        return self.classifier.predict(rep)
    def to_json(self):
        rtn = {"locale" : self.language}
        rtn["version"] = PIPELINE_VERSION
        rtn["timestamp"] = str( datetime.utcnow() )
        rtn["transformations"] = []
        for t in self.representation:
            rtn["transformations"].append(t.to_json()) 
        rtn["classifier"] = self.classifier.to_json()
        return rtn
    def save(self, filename):
        tmp = self.to_json()
        with open(filename, 'w') as f:
            f.write(json.dumps(tmp))

def load_model(filename):
    pass
