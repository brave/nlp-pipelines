from transformation import apply_transform
from classifier import Classifier, Classifier_Type, classifier_from_proto
from enum import Enum
from datetime import datetime
import json
from pipeline_pb2 import Pipeline

PIPELINE_VERSION = 0

class Language(Enum):
    EN = "EN"
    DE = "DE"
    FR = "FR"
    JA = "JA"

class NLP_Model:
    def __init__(self, language, representation=None, 
                classifier_type=Classifier_Type.NB, version = PIPELINE_VERSION, 
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
            tmp = apply_transform(transform, tmp)
        return tmp
    def train(self, data, labels):
        rep = data
        if self.representation is not None:
            rep = self.apply_transforms(data)
        self.classifier.train(rep,labels)
    def predict(self, data):
        rep = data
        if self.representation is not None:
            rep = self.representation.apply_transforms(data)
        return self.classifier.predict(rep)
    def to_proto(self):
        return Pipeline(version = self.version, language = self.language, 
                timestamp = str(datetime.utcnow()), representation = self.representation,
                classifier = self.classifier)

def model_from_proto(model_proto):
    return NLP_Model(version = model_proto.version, 
                    language=model_proto.language, 
                    representation=model_proto.representation,
                    classifier = classifier_from_proto(model_proto.classifier))