import pandas as pd 
import numpy as np
from importlib import reload
import nltk
import spacy
import en_core_sci_lg # Biomedical word embeddings
from utils import clean_text, save_pickle, load_pickle
from nltk.corpus import stopwords as _stopwords
from tqdm import tqdm

import os
import re 
from string import punctuation as PUNCTUATION
from nltk.corpus import stopwords as _stopwords
from gensim.models import Word2Vec
from os.path import join as join_path
from gensim.models.callbacks import CallbackAny2Vec
from coordle_backend import CordDoc, Index
import sys 
from time import time 

class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, output_dir: str, prefix: str, logs_filename: str):
        self.output_dir = output_dir
        self.prefix = prefix
        self.logs_filename = logs_filename
        self.epoch = 0

    def on_epoch_end(self, model):
        cum_loss = model.get_latest_training_loss()
        if self.epoch == 0:
            loss = cum_loss
        else:
            loss = cum_loss - self.loss_previous_step
        self.loss_previous_step = loss
        with open(join_path(self.output_dir, self.logs_filename), 'a+') as file:
            file.write(f'Epoch #{self.epoch}, loss: {loss}\n')
        
        output_path = join_path(self.output_dir, f'{self.prefix}_epoch_{self.epoch}.model')
        model.save(output_path)
        self.epoch += 1    

if __name__ == '__main__':
    option = sys.argv.pop(0)

    df = pd.read_csv('data/cord-19-data.csv')
    # Load the last trained model
    print('Loading word2vec weights')
    sys.stdout.flush()
    model = Word2Vec.load(join_path('models-word2vec', 'w2v_model_epoch_29.model'))
    word_to_int = {word:i for i, word in enumerate(model.wv.index2word)}
    int_to_word = np.array(model.wv.index2word)
    W = model.trainables.syn1neg

    print('Created index')
    sys.stdout.flush()
    coordle = Index(W, int_to_word, word_to_int)

    print('Appending to index')
    sys.stdout.flush()
    t0 = time()
    # for i in range(1024):
    for i in range(len(df)):
        coordle.add(df.iloc[i]['cord_uid'], df.iloc[i]['title'], df.iloc[i]['body_text'])
        if i % 100 == 0:
            print(f'At iteration {i}')
            sys.stdout.flush()
    print(f'Indexing files took {time()-t0:.2f} seconds')
    # filename='coordle/coordle_index_lite.p'
    filename='coordle/coordle_index_full.p'
    sys.stdout.flush()
    print(f'Dumping index object to {filename}')
    save_pickle(coordle, filename)