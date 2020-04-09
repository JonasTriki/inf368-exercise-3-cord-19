# Imports
from time import time
from os.path import join as join_path
import numpy as np
import pandas as pd

import multiprocessing
cores = multiprocessing.cpu_count()

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
import logging # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

import nltk
nltk.download('punkt')

from utils import clean_text

from sklearn.metrics.pairwise import cosine_similarity
import sys 

def flusher(f):
    def wrap(*args, **kwargs):
        f(*args, **kwargs)
        sys.stdout.flush()
    return wrap

print = flusher(print)

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

class CORDDataIteratorWord2Vec():
    def __init__(self, texts: np.ndarray):
        self.texts = texts
    
    def __iter__(self):
        for text in self.texts:
            sentences = nltk.tokenize.sent_tokenize(text)
            cleaned_sentences = [clean_text(sent).split() for sent in sentences]
            for sentence in cleaned_sentences:
                yield sentence

class CORDDataIteratorDoc2Vec():
    def __init__(self, texts: np.ndarray):
        self.texts = texts
    
    def __iter__(self):
        for cord_uid, cord_text in self.texts:
            sentences = nltk.tokenize.sent_tokenize(cord_text)
            cleaned_sentences = [clean_text(sent) for sent in sentences]
            for sentence in cleaned_sentences:
                yield TaggedDocument(sentence, [cord_uid])
if __name__ == '__main__':
    # Constants
    cord_data_dir = 'data'
    cord_data_path = join_path(cord_data_dir, 'cord-19-data.csv')
    w2v_saved_models_dir = 'models-word2vec-2'
    # d2v_saved_models_dir = 'models-doc2vec'
    saved_models_prefix = 'model'
    train_logs_path = 'train_logs.txt'

    cord_data = pd.read_csv(cord_data_path)
    cord_data_eng = cord_data[cord_data['language'] == 'en']
    eng_texts = cord_data_eng[['cord_uid', 'body_text']].values

    print('Getting senctences')
    cord_num_sentences = 0
    for _, text in eng_texts:
        sentences = nltk.tokenize.sent_tokenize(text)
        cord_num_sentences += len(sentences)
    print(f'Total number of CORD-19 sentences: {cord_num_sentences}')

    cord_sentences = CORDDataIteratorDoc2Vec(eng_texts)

    print('Setting up w2v model')
    # Setup initial model
    w2v_model = Word2Vec(
        min_count=20,
        window=2,
        size=300,
        sample=1e-5,
        alpha=0.025,
        min_alpha=0.0005,
        negative=20,
        workers=cores-1,
        callbacks=[EpochSaver(w2v_saved_models_dir, saved_models_prefix, train_logs_path)]
    )

    print('Building vocan')
    # Build vocabulary
    t = time()
    w2v_model.build_vocab(cord_sentences, progress_per=int(cord_num_sentences / 100))
    print(f'Time to build vocab: {round((time() - t) / 60, 2)} mins')

    print('Train model')
    # Train model
    t = time()
    w2v_model.train(cord_sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=30)
    print(f'Time to train the model: {round((time() - t) / 60, 2)} mins')
    print('Done!')