from classifier import Classifier, Classifier_Type
from enum import Enum

class Language(Enum):
    EN = "EN"
    DE = "DE"
    FR = "FR"
    JA = "JA"

class NLP_Representation:
    ''' Poor man's pipeline implementation to keep things somewhat consistent and 
        reproducible in NLP pipelines that can be ported to C++ 
    '''
    def __init__(self, transforms):
        self.transforms = transforms
    def apply_transforms(self, texts):
        ''' linear apply operation'''
        last_step = texts
        for transform in self.transforms:
            this_step = transform.apply_transform(last_step)
            last_step = this_step
        return last_step

class NLP_Model:
    def __init__(self, language, representation=None, classifier_type=Classifier_Type.NB):
        if language not in Language.__members__:
            raise ValueError('Unknown language')
        self.representation=representation
        self.classifier = Classifier(classifier_type=classifier_type)
    def train(self, data, labels):
        rep = data
        if self.representation is not None:
            rep = self.representation.apply_transforms(data)
        self.classifier.train(rep,labels)
    def predict(self, data):
        rep = data
        if self.representation is not None:
            rep = self.representation.apply_transforms(data)
        return self.classifier.predict(rep)