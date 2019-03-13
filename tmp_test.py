from transformation import Transformation, Transformation_Type
from nlp_pipeline import NLP_Representation, NLP_Model
from classifier import classifier_from_json
import json 

EPS = 1e-4

def setup_data():
    train_messages = ['This is a spam email.', 
                 'Another spam trying to sell you viagra',
                 'Message from mom with no real subject', 
                 'Another messase from mom with no real subject',
                 'Yadayada']
    train_labels = ['spam', 'spam', 'ham', 'ham', 'junk']

    test_messages = ['Even more spam because we love selling you viagra',
                'Dont forget to grab a jacket, love mom',
                'yada!']
    test_labels = ['spam', 'ham','junk']
    return (train_messages, train_labels, test_messages, test_labels)

# Make sure we can train and predict reasonably well with a pipeline 
def test_train_test_python():
    train_messages, train_labels, test_messages, test_labels = setup_data()
    nlp_model = NLP_Model(
    representation=NLP_Representation([Transformation(transformation_type=Transformation_Type.TO_LOWER),
                            Transformation(transformation_type=Transformation_Type.HASHED_NGRAMS,
                            params={'n_range':[1,2,3,4,5], 'num_buckets':500}) ]), 
    classifier_type = 'NB', 
    language = 'EN')
    nlp_model.train(train_messages, train_labels)
    preds_train = nlp_model.predict(train_messages)
    preds_test = nlp_model.predict(test_messages)
    for i, pred in enumerate(preds_train):
        assert (pred==train_labels[i])
    for i, pred in enumerate(preds_test):
        assert (pred==test_labels[i])
# make sure we can save and load a model with presision within epsilon(EPS)
def test_save_load():
    train_messages, train_labels, test_messages, test_labels = setup_data()
    nlp_model = NLP_Model(
    representation=NLP_Representation([Transformation(transformation_type=Transformation_Type.TO_LOWER),
                            Transformation(transformation_type=Transformation_Type.HASHED_NGRAMS,
                            params={'n_range':[1,2,3,4,5], 'num_buckets':500}) ]), 
    classifier_type = 'NB', 
    language = 'EN')
    nlp_model.train(train_messages, train_labels)
    nlp_model.save('test_model.json')
    with open('test_model.json','r') as f:
        tmp_model=f.read()
    restored_model = json.loads(tmp_model)
    assert restored_model['language'] == nlp_model.language
    classifier_json=json.loads(restored_model['classifier'])
    for i, k in enumerate(classifier_json['class_weights']):
        assert k in nlp_model.classifier.name_to_labels
        for j, w in enumerate(classifier_json['class_weights'][k]):
            assert nlp_model.classifier.classifier.coef_[i,j]-w <=EPS

def test_train_test_json():
    train_messages, train_labels, test_messages, test_labels = setup_data()
    nlp_model = NLP_Model(
    representation=NLP_Representation([Transformation(transformation_type=Transformation_Type.TO_LOWER),
                            Transformation(transformation_type=Transformation_Type.HASHED_NGRAMS,
                            params={'n_range':[1,2,3,4,5], 'num_buckets':500}) ]), 
    classifier_type = 'NB', 
    language = 'EN')
    nlp_model.train(train_messages, train_labels)

    rep_train = nlp_model.representation.apply_transforms(train_messages)
    rep_test = nlp_model.representation.apply_transforms(test_messages)
    restored_classifier=classifier_from_json(nlp_model.classifier.to_json())
    preds_train = restored_classifier.predict(rep_train)
    preds_test = restored_classifier.predict(rep_test)
    for i, pred in enumerate(preds_train):
        assert (pred==train_labels[i])
    for i, pred in enumerate(preds_test):
        assert (pred==test_labels[i])
