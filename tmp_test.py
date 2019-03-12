from transformation import Transformation, Transformation_Type
from nlp_pipeline import NLP_Representation, NLP_Model

EPS = 1e-5

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
nlp_model = NLP_Model(
    representation=NLP_Representation([Transformation(transformation_type=Transformation_Type.TO_LOWER),
                            Transformation(transformation_type=Transformation_Type.HASHED_NGRAMS,
                            params={'n_range':[1,2,3,4,5], 'num_buckets':500}) ]), 
    classifier_type = 'NB', 
    language = 'EN')
    
    
nlp_model.train(train_messages, train_labels)
nlp_model.predict(train_messages)
nlp_model.predict(test_messages)
