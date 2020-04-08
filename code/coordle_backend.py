import numpy as np
import pandas as pd
from utils import clean_text
from typing import Iterable, Union, Callable
import nltk
from collections.abc import Iterable
from itertools import chain
import pickle
from copy import deepcopy

class SentVectorDoc:
    '''
    Class for documents.

    Will calculate vectors for each sentence in text.
    '''
    def __init__(self, W: np.ndarray, int_to_word: np.ndarray,
                 word_to_int: dict, uid: str, title: str=None,
                 store_sents: bool=False):
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
        # Word2Vec array
        self.W = W
        self.int_to_word = int_to_word
        self.word_to_int = word_to_int
        self.uid = uid

        self.title = title
        self.store_stringsents = store_sents

        # Stored sentences, will be used if self.store_sents is True
        self.stringsents: list = None
        self.wordfreqs = dict()

        self._tf_idf_score: float = 0

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        return self.uid

    def __str__(self):
        s = f'uid: {self.uid}\n'
        s += f'title: {self.title}'
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

    def fit(self, text: str, cleaner: Callable=None, **kwargs):
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

        # stringsents = nltk.tokenize.sent_tokenize(self.text)
        # if self.store_stringsents:
            # self.stringsents = stringsents

        # tokenized_sents is a list of lists as cleaner function
        # should return a list of cleaned tokens.
        # This is the slowest part of the code
        # tokenized_sents = [cleaner(s, **kwargs) for s in stringsents]
        tokens = cleaner(text)
        uniques, counts = np.unique(tokens, return_counts=True)
        self.wordfreqs = {word:count for word, count in zip(uniques, counts)}

        # Initalize memory for sentence vectors
        # self.sentvecs = np.zeros((len(tokenized_sents), self.W.shape[1]))

        # To use as default value if word_to_int does not contain a token
        # origo = np.zeros(self.W.shape[1])

        # Calculate vectors for sentences by adding up the word vectors
        # for each sentence
        # for i, sentence_tokens in enumerate(tokenized_sents):
        #     for token in sentence_tokens:
        #         if token in self.word_to_int:
        #             # Get wordvec
        #             wordvec = self.W[int(self.word_to_int[token])]
        #             # Need word frequencies to calculate TF-IDF
        #             if token not in self.wordfreqs:
        #                 self.wordfreqs[token] = 0
        #             self.wordfreqs[token] += 1
        #         else:
        #             # Unknown tokens are considered as origo
        #             wordvec = origo
        #         # Build sentence vector
        #         self.sentvecs[i] += wordvec

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
    def __init__(self, W: np.ndarray, int_to_word: np.ndarray,
                 word_to_int: dict, name: str='Coordle'):
        self.W = W
        self.int_to_word = int_to_word
        self.word_to_int = word_to_int
        self.docmap = dict()
        self.uid_docmap = dict()

    def __len__(self):
        return len(self.docmap)
    
    def __getitem__(self, query: str):
        '''
        Implements fancy syntax: coordle['query here']
        '''
        return self.search(query)

    def add(self, uid: str, title: str, text: str):
        '''
        Adds document to index

        Parameters:
        -------------
        uid: unique identification string

        title: title of document

        text: text of document
        '''
        doc = SentVectorDoc(
            W=self.W,
            int_to_word=self.int_to_word,
            word_to_int=self.word_to_int,
            uid=uid,
            title=title
        )
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


    def tf_idf(self, result: set, token: str):
        for doc in result:
            idf = np.log(len(self.docmap) / len(result))
            tf = doc.wordfreqs[token] / len(doc)
            doc._tf_idf_score += tf*idf

    def clear_tf_idf(self, docs: list):
        pass

    def search(self, query: str) -> tuple:
        '''
        Returns a list of query results given
        query as string
        '''
        querytokens = clean_text(query)
        print('Query tokens: ', querytokens)

        init_token = querytokens.pop(0)
        if init_token in self.docmap:
            results = self.docmap[init_token]

        # TODO: Make class for query result or something
        # since scores can get fucked up, also improve
        # readability
        for token in querytokens:
            if token in self.docmap:
                result: set = self.docmap[token]
                # Calculates tf_idf and saves them in object attributes
                self.tf_idf(result, token)
                # Update with union
                results |= result
        
        results_list = list(results)
        results_list_cp = deepcopy(list(results))
        
        # Insanity check
        # for result, result_cp in zip(results_list, results_list_cp):
        #     print(f'id: {id(result)}, {id(result_cp)}')
        #     print(f'is: {result is result_cp}')
        #     print(f'eq: {result == result_cp}')
        #     print(f'vecs: {result.sentvecs is result_cp.sentvecs}')
        #     print(f'wordfreqs: {result.wordfreqs is result_cp.wordfreqs}')
        #     print()

        # Reset _tf_idf_score counter
        for result in results_list:
            result._tf_idf_score = 0

        # Sort 
        results_list_cp = sorted(
            results_list_cp, 
            key=lambda x: x._tf_idf_score,
            reverse=True
        )
        scores = [result._tf_idf_score for result in results_list_cp]

        return results_list_cp, scores

        # print(self.docmap['white'])

