import numpy as np
import pandas as pd
from utils import clean_text
from typing import Iterable, Union, Callable
import nltk
from collections.abc import Iterable
from itertools import chain
import pickle
from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm
from os import cpu_count

class SentVectorDoc:
    '''
    Class for documents.

    Will calculate vectors for each sentence in text.
    '''
    def __init__(self, uid: str, title: str=None):
        '''
        Parameters
        -----------
        W: Word2Vec 2D array

        uid: unique id of document

        int_to_word: Arry of words, where indices corresponds to the word_to_int
                    dictionary

        word_to_int: Dictionary where keys are words, and values are integers

        uid: unique string that represents the document, stands for unique
             identity.

        title: optional, title of document

        store_sents: Whether to store string sentences is memory or not.
                     Will be stored self.stringsents
        '''
        self.uid = uid
        self.title = title
        self.wordfreqs = dict()
        self._tf_idf_score: float = 0

    def __len__(self):
        '''
        Returns length of text that was trained on 
        '''
        return self.len

    def __repr__(self):
        return self.uid

    def __str__(self):
        s = f'uid: {self.uid}\n'
        s += f'title: {self.title[:12]}'
        return s

    def __hash__(self):
        '''
        SentVectorDoc objects are identified by uid
        '''
        return hash(self.uid)

    def __eq__(self, other):
        '''
        SentVectorDoc objects are identified by uid
        '''
        return hash(self.uid) == hash(other.uid)

    def fit(self, text: str, cleaner: Callable=None, **kwargs) -> tuple:
        '''
        Create sentence vectors for all sentences in given text.
        (WIP) OUTDATED DOCSTRING!
        Parameters:
        text: Text as string

        cleaner: Text cleaning function, should take in a string as first
                 argument, and return an iterable of cleaned tokens from string.
                 If None is given,defaults to clean_text from bundled utils
                 module

        kwargs: Keyword arguments for cleaner function

        Returns
        --------
        Tuple of self and the cleaned and tokenized sentences
        (self, unique_tokens)
        '''
        self.len = len(text)

        if type(text) == str:
            if cleaner is None:
                cleaner = clean_text
            tokens = cleaner(text)
        elif isinstance(text, Iterable):
            tokens = text
        else:
            raise ValueError(f'Unsupported type for text, got f{type(text)}')

        uniques, counts = np.unique(tokens, return_counts=True)
        self.wordfreqs = {word:count for word, count in zip(uniques, counts)}
        return self, uniques

    def fit_old(self, text: str, cleaner: Callable=None, **kwargs):
        '''
        Create sentence vectors for all sentences in given text.
        (WIP) OUTDATED DOCSTRING!
        Parameters:
        text: Text as string

        cleaner: Text cleaning function, should take in a string as first
                 argument, and return an iterable of cleaned tokens from string.
                 If None is given,defaults to clean_text from bundled utils
                 module

        kwargs: Keyword arguments for cleaner function

        Returns
        --------
        Tuple of self and the cleaned and tokenized sentences
        (self, tokenized_sents)
        '''
        if not type(text) == str:
            raise ValueError(f'Unsupported type for text, got f{type(text)}')

        self.text = text

        if cleaner is None:
            cleaner = clean_text

        stringsents = nltk.tokenize.sent_tokenize(self.text)
        if self.store_stringsents:
            self.stringsents = stringsents

        # tokenized_sents is a list of lists as cleaner function
        # should return a list of cleaned tokens.
        # This is the slowest part of the code
        tokenized_sents = [cleaner(s, **kwargs) for s in stringsents]

        # Initalize memory for sentence vectors
        self.sentvecs = np.zeros((len(tokenized_sents), self.W.shape[1]))

        # To use as default value if word_to_int does not contain a token
        origo = np.zeros(self.W.shape[1])

        # Calculate vectors for sentences by adding up the word vectors
        # for each sentence
        for i, sentence_tokens in enumerate(tokenized_sents):
            for token in sentence_tokens:
                if token in self.word_to_int:
                    # Get wordvec
                    wordvec = self.W[int(self.word_to_int[token])]
                    # Need word frequencies to calculate TF-IDF
                    if token not in self.wordfreqs:
                        self.wordfreqs[token] = 0
                    self.wordfreqs[token] += 1
                else:
                    # Unknown tokens are considered as origo
                    wordvec = origo
                # Build sentence vector
                self.sentvecs[i] += wordvec
        return self, tokenized_sents

class Index:
    # TODO: Inspect why query is slow
    def __init__(self):
        self.docmap = dict()
        self.uid_docmap = dict()

    def __len__(self):
        return len(self.docmap)
    
    def __getitem__(self, query: str) -> tuple:
        '''
        Implements fancy syntax: coordle['query here']
        '''
        return self.search(query)

    def add(self, uid: str, title: str, text: Union[str, Iterable]):
        '''
        Adds document to index

        Parameters:
        -------------
        uid: unique identification string

        title: title of document

        text: text of document
        '''
        doc = SentVectorDoc(uid=uid, title=title)
        doc, unique_tokens = doc.fit(text)

        # _________old____________
        # doc, tokens = doc.fit(text)
        # # Fast way to flatten Python list of lists
        # tokens = list(chain.from_iterable(tokens))
        # unique_tokens = np.unique(tokens) 
        # _________________________

        # Add document to hasmap where keys are uids and values are docs
        self.uid_docmap[doc.uid] = doc

        # Add document to hashmap where keys are unique tokens, and values
        # are sets
        for token in unique_tokens:
            if token not in self.docmap:
                self.docmap[token]=set()
            self.docmap[token].add(doc)       

    def build_from_df(self, df: pd.DataFrame, uid: str, title: str, 
                      text: str, use_multiprocessing: bool=False, 
                      workers: int=1, verbose: bool=True, 
                      cleaner: Callable=None):
        '''
        Build index given pd.DataFrame
        '''
        if cleaner is None:
            cleaner = clean_text

        tqdm_args = {'total':len(df), 'position':0, 'disable':not verbose}

        if use_multiprocessing:
            if workers == -1:
                workers = cpu_count()

            if verbose:
                print(f'Text cleaning initilized on {workers} workers')
            with Pool(workers) as pool:
                # texts=pool.map(cleaner, df[text])
                clean_iterator = tqdm(
                    pool.imap(cleaner, df[text]), 
                    desc='Cleaning texts', 
                    **tqdm_args    
                )
                texts=list(clean_iterator)
        else:
            texts = df[text]
        
        uids = df[uid]
        titles = df[title]

        for uid_, title_, text_ in tqdm(zip(uids, titles, texts), 
                                        desc='Adding to index', **tqdm_args):
            self.add(uid_, title_, text_)

    def _tf_idf(self, result: set, token: str):
        for doc in result:
            idf = np.log(len(self.docmap) / len(result))
            tf = doc.wordfreqs[token] / len(doc)
            doc._tf_idf_score += tf*idf

    def get_doc(self, uid: str):
        '''
        Get document given uid
        '''
        return self.uid_docmap[uid]

    def search(self, query: Union[str, list], verbose=False) -> tuple:
        '''
        Returns a list of query results given
        query as string or list of strings, also returns tf-idf scores
        '''
        if type(query) == str:
            querytokens = clean_text(query)
        elif type(query) == list:
            querytokens = query
        else:
            raise ValueError(f'query must be of type string or list, got {type(query)}')

        results = set()

        if verbose: print('Query tokens: ', querytokens)
        init_token = querytokens.pop(0)
        if init_token in self.docmap:
            results = self.docmap[init_token]

        for token in querytokens:
            if token in self.docmap:
                result: set = self.docmap[token]
                # Calculates tf_idf and saves them in object attributes
                # hence no return value
                self._tf_idf(result, token)
                # Update with union
                results |= result
        
        if results is None:
            if verbose: print('No results')
            return [], []

        results_list = list(results)
        results_list.sort(key=lambda x: x._tf_idf_score, reverse=True)
        
        scores = np.array([result._tf_idf_score for result in results_list])
        scores = scores/scores.sum()*100

        # Reset _tf_idf_score counter on things in index
        for result in results_list:
            result._tf_idf_score = 0

        return results_list, scores

class AI_Index(Index):
    '''
    Essentially, uses TF-IDF, but adds similar query tokens 
    to given query using AI, Big Data and Machine Learning $$$
    '''
    def __init__(self, similar: Callable, n_similars: int=3):
        super().__init__()
        self.similar = similar
        self.n_similars = n_similars
    
    def _append_most_similar_tokens(self, tokens: list):
        '''
        Given a list of tokens, for each token in tokens, append the 
        most similar 
        '''
        for token in reversed(tokens):
            if token in self.model.wv:
                similars = [word for word, _ in \
                    self.similar(token)[:self.n_similars]]
                tokens.extend(similars)    
    
    def _get_query_tokens(self, query: Union[str, list]):
        '''
        Assert query type, and do appropriate preprocessing
        '''
        if type(query) == str:
            querytokens = clean_text(query)
        elif type(query) == list:
            querytokens = query
        else:
            raise ValueError(
                f'query must be of type string or list, got {type(query)}')
        return querytokens

    def search(self, query: Union[str, list], verbose=False):
        '''
        Given a query, obtain query tokens by using clean_text(query).
        Then for each query token, get the top n most similar tokens and 
        append them to the query. Then do regular search and score relevance
        with TF-IDF.

        Pros: Easy integration with the regular index, relevance scoring 
              becomes trivial. Very reasonable thing to do IMO. 

        Cons: Not that ambitious maybe? 
        '''
        querytokens = self._get_query_tokens(query)
        self._append_most_similar_tokens(querytokens) # In place
        return super().search(querytokens, verbose)
    