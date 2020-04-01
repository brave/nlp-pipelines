from nlp_pipelines.transformation import To_lower, Hashed_ngrams
from nlp_pipelines.nlp_pipeline import NLP_Model, load_model
from nlp_pipelines.transformation import clean_texts
# import json 

EPS = 1e-6

def setup_data():
    train_messages = ['This is a spam email.', 
                 'Another spam trying to sell you viagra',
                 'Message from mom with no real subject', 
                 'Another message from mom with no real subject',
                 'Yadayada']
    train_labels = ['spam', 'spam', 'ham', 'ham', 'junk']

    test_messages = ['Even more spam because we love selling you viagra',
                'Did you get my message? love mom',
                'yada!']
    test_labels = ['spam', 'ham','junk']
    return (train_messages, train_labels, test_messages, test_labels)
def setup_hashing_model():
    to_lower = To_lower()
    hashed_ngrams = Hashed_ngrams(n_range=[4], num_buckets=1000)
    return NLP_Model(language='EN', representation=[to_lower, hashed_ngrams], 
                    classifier_type = 'LINEAR')


# Make sure we can train and predict reasonably well with a pipeline 
def test_train_test_python():
    train_messages, train_labels, test_messages, test_labels = setup_data()
    nlp_model = setup_hashing_model()
    nlp_model.train(train_messages, train_labels)
    preds_train = nlp_model.predict(train_messages)
    preds_test = nlp_model.predict(test_messages)
    for i, pred in enumerate(preds_train):
        assert (pred==train_labels[i])
    for i, pred in enumerate(preds_test):
        assert (pred==test_labels[i])

def test_text_cleaning():
    gold_test_in = '''The quick brown fox jumps over the lazy dog. $123,000.0 !\"#$%&'()*+,-./:<=>?@\\[]^_`{|}~ 0123456789 \t\n\v\f\r 0x7F x123x a1b2c3 Les naïfs ægithales hâtifs pondant à Noël où il gèle sont sûrs d'être déçus en voyant leurs drôles d'œufs abîmés. Falsches Üben von Xylophonmusik quält jeden größeren Zwerg. ξεσκεπάζω την ψυχοφθόρα \\t\\n\\v\\f\\r βδελυγμία. いろはにほへど　ちりぬるを わがよたれぞ　つねならむ うゐのおくやま　けふこえて あさきゆめみじ　ゑひもせず '''
    gold_test_out = '''The quick brown fox jumps over the lazy dog Les naïfs ægithales hâtifs pondant à Noël où il gèle sont sûrs d être déçus en voyant leurs drôles d œufs abîmés Falsches Üben von Xylophonmusik quält jeden größeren Zwerg ξεσκεπάζω την ψυχοφθόρα βδελυγμία いろはにほへど　ちりぬるを わがよたれぞ　つねならむ うゐのおくやま　けふこえて あさきゆめみじ　ゑひもせず'''
    cleaned = clean_texts([gold_test_in])[0]
    assert(cleaned == " ".join(gold_test_out.split() ) )
# make sure we can save and load a model with presision within epsilon(EPS)
# def test_save_load():
#     train_messages, train_labels, test_messages, test_labels = setup_data()
#     nlp_model = setup_hashing_model()
#     nlp_model.train(train_messages, train_labels)
#     nlp_model.save('test_model.pb')
#     restored_model = load_model('test_model.pb')
#     assert restored_model.language == nlp_model.language
#     # classifier_json=json.loads(restored_model['classifier'])
#     for i, k in enumerate(restored_model.classifier.class_weights.keys()):
#         assert k in nlp_model.classifier.name_to_labels
#         for j, w in enumerate(restored_model.classifier.class_weights[k]):
#             assert nlp_model.classifier.classifier.coef_[i,j]-w <=EPS

# def test_train_test_json():
#     train_messages, train_labels, test_messages, test_labels = setup_data()
#     nlp_model = setup_hashing_model()
#     nlp_model.train(train_messages, train_labels)
#     rep_train = nlp_model.apply_transforms(train_messages)
#     rep_test = nlp_model.apply_transforms(test_messages)
#     #restored_classifier=classifier_from_proto(nlp_model.classifier.to_proto())
#     preds_train = restored_classifier.predict(rep_train)
#     preds_test = restored_classifier.predict(rep_test)
#     for i, pred in enumerate(preds_train):
#         assert (pred==train_labels[i])
#     for i, pred in enumerate(preds_test):
#         assert (pred==test_labels[i])
