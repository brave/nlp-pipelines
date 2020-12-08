from datetime import datetime
from enum import Enum
import json
import os

from .classifier import Classifier, Classifier_Type

PIPELINE_VERSION = 1

class Language(Enum):
    EN = 'EN'
    DE = 'DE'
    FR = 'FR'
    JA = 'JA'
    ES = 'ES'
    PT = 'PT'
    EL = 'EL'
    AR = 'AR'
    FI = 'FI'
    HE = 'HE'
    HI = 'HI'
    IT = 'IT'
    KO = 'KO'
    NL = 'NL'
    PL = 'PL'
    RO = 'RO'
    RU = 'RU'
    SV = 'SV'
    TL = 'TL'
    TR = 'TR'
    ZH = 'ZH'
    
Language_to_code = {
    'EN' : 'emgmepnebbddgnkhfmhdhmjifkglkamo',
    'FR' : 'hbejpnagkgeeohiojniljejpdpojmfdp',
    'DE' : 'eclclcmhpefndfimkgjknaenojpdffjp',
    'JA' : 'ncnmgkcadooabjhgjlkkdipdnfokpjnm',
    'PT' : 'ikpplkdenofcphgejneekjmhepajgopf',
    'ES' : 'ahiocclicnhmiobhocikfdamfccbehhn',
    'AR' : 'hfiknbegiiiigegdgpcgekhdlpdmladb',
    'EL' : 'aefhgfnampgebnpchhfkaoaiijpmhcca',
    'FI' : 'djokgcimofealcnfijnlfdnfajpdjcfg',
    'HE' : 'gffjpkbdngpbfkflpnoodjfkpelbappk',
    'HI' : 'emhbebmifclalgbdpodobmckfehlkhfp',
    'IT' : 'ijaiihoedhaocihjjkfjnhfhbceekdkg',
    'KO' : 'jbhiacghlejpbieldkdfkgenhnolndlf',
    'NL' : 'opoleacilplnkhobipjcihpdoklpnjkk',
    'PL' : 'aijecnhpjljblhnogamehknbmljlbfgn',
    'RO' : 'hffipkehifobjlkdjagndofmpjnpkgje',
    'RU' : 'nigmjcnboijpcoikglccmoncigioojpa',
    'SV' : 'jpgndiehmchkacbfggdgkoohioocdhbp',
    'TL' : 'kcoilhabhhnfdakenmhddnhngngggcmp',
    'TR' : 'kpdcfihnokkbialolpedfamclbdlgopi',
    'ZH' : 'oblfikajhadjnmjiihdchdfdcfehlbpj'
}

def build_manifest(country_string):
    manifest = {}
    manifest['schemaVersion'] = PIPELINE_VERSION
    manifest['models'] = [{}]
    manifest['models'][0]['id'] = country_string
    manifest['models'][0]['filename'] = country_string
    manifest['models'][0]['version'] = PIPELINE_VERSION
    return json.dumps(manifest)

class NLP_Model:
    def __init__(self,
                 language,
                 representation=None,
                 classifier_type=Classifier_Type.LINEAR,
                 version=PIPELINE_VERSION,
                 classifier=None,
                 reg_param=1.0):
        if language.upper() not in Language.__members__:
            raise ValueError('Unknown language')
        self.version = version
        self.language = language
        self.representation = representation
        if classifier is None:
            self.classifier = Classifier(classifier_type=classifier_type, reg_param=reg_param)
        else:
            self.classifier = classifier

    def apply_transforms(self, data):
        tmp = data
        for transform in self.representation:
            tmp = transform.apply(tmp)
        return tmp

    def train(self, data, labels, positive_class=None):
        rep = data
        if self.representation is not None:
            rep = self.apply_transforms(data)
        self.classifier.train(rep, labels, positive_class)

    def partial_train(self, data, labels, uniq_labels):
        rep = data
        if self.representation is not None:
            rep = self.apply_transforms(data)
        self.classifier.partial_train(rep,labels, uniq_labels)

    def predict(self, data):
        rep = data
        if self.representation is not None:
            rep = self.apply_transforms(data)
        return self.classifier.predict(rep)

    def to_json(self):
        rtn = {'locale': self.language}
        rtn['version'] = PIPELINE_VERSION
        rtn['timestamp'] = str(datetime.utcnow())
        rtn['transformations'] = []
        for t in self.representation:
            rtn['transformations'].append(t.to_json())
        rtn['classifier'] = self.classifier.to_json()
        return rtn

    def save(self, output_dir):
        country_string = Language_to_code[self.language]
        tmp = self.to_json()
        manifest = build_manifest(country_string)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, country_string), 'w') as f:
            f.write(json.dumps(tmp))
        with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
            f.write(manifest)

def load_model(filename):
    with open(filename,'r') as f:
        json_pipeline = json.loads(f.read())
    json_transformations = json_pipeline['transformations']
    json_classifier = json_pipeline['classifier']
