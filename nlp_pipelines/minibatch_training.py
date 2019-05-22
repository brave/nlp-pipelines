#simple minibatch training procedures for online sgd pipelines with hinge loss in sklearn 
import numpy as np

def minibatch_loader(pipeline, texts,labels, epochs = 5, batch_size=128):
    is_init=False
    uniq_labels = list(set(labels))
    print("uniq_labels = ", uniq_labels)
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        idx = np.random.permutation(len(labels))
        batch_start = 0
        while batch_start<len(labels):
            this_idx = idx[batch_start: (batch_start+batch_size)]
            X = [texts[i] for i in this_idx]
            Y = [labels[i] for i in this_idx]
            pipeline.partial_train(X,Y, uniq_labels)
            batch_start += batch_size
            print('.', end='')
        print('*')
