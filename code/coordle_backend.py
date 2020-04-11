import numpy as np
import pandas as pd
from utils import clean_text
from typing import Iterable, Union, Callable
import nltk
from collections import deque
from collections.abc import Iterable
from itertools import chain
import pickle
from copy import deepcopy
from multiprocessing import Pool
from tqdm import tqdm
from os import cpu_count
from string import punctuation as PUNCTUATION
import re

class SentVectorDoc:
    '''
    Class for documents.
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
        return self.uid == other.uid
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __contains__(self, word):
        return word in self.wordfreqs

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


class RecursiveDescentParser:
    '''
    Parser class for Coordle used for parsing queries 
    and also searching using the parsed queries. 

    Made to parse query tokens contained 
    in deque objects.

    This class is made to be composed in coordle_backend.Index
    '''
    def __init__(self, token_to_set: dict, or_operator: str='OR', 
                 and_operator: str='AND', difference_operator: str='NOT', 
                 punctuation: str=None):
        '''
        Parameters:
        ------------
        sets: a dictionary where keys correspond to query tokens, and values
              are sets containing SentVectorDoc objects
        '''
        self.token_to_set = token_to_set
        self.n_valid_tokens: int = len(self.token_to_set)
        
        self.or_operator = or_operator
        self.and_operator = and_operator
        self.difference_operator = difference_operator

        self.operators = {
            self.or_operator,
            self.and_operator,
            self.difference_operator
        }

        self.punctuation = punctuation
        if self.punctuation is None:
            self.punctuation = PUNCTUATION.replace('(','').replace(')','')

    def get_logical_querytokens(self, query: str):
        query = re.sub(f'[{self.punctuation}]','',query)
        querytokens = re.split('([^a-zA-Z0-9])', query)
        # Gotta do this to capture parenthesis
        querytokens = chain.from_iterable([t.split() for t in querytokens])

        q1 = deque(querytokens)
        q2 = deque()
   
        q2.append(q1.popleft())

        while len(q1) > 0:
            token = q1.popleft()
            
            if q2[-1] == '(' or token == ')':
                q2.append(token)
                continue
            
            # If preceeding token was operator
            if q2[-1] in self.operators:
                q2.append(token)
            # If preceeding token was not operator
            else:
                # If current token is not an operator
                if token not in self.operators:
                    q2.append(self.or_operator)
                q2.append(token)
        return q2

    def assert_query(self, querytokens: deque, errmsgs: list) -> bool:
        '''
        Check if query is properly formatted. Returns True if everything is ok,
        else False. 
        '''
        q = querytokens.copy()
        p_list = []
        p_counter = 0 
        flag = True
        curr = q.popleft()
        
        ##### Initialize by checking the first token #####
        
        # Stray closing parenthesis
        if curr == ')':
            p_counter -= 1
            p_list.append(p_counter)
            flag = False
        
        if curr == '(':
            p_counter += 1
            p_list.append(p_counter)
        
        # If query is starting with an operator
        if curr in self.operators:
            errmsgs.append(f'SyntaxError: First token "{curr}" is an operator')
            flag = False
        
        # If querytokens consisted of only a single token
        if len(q) == 0:
            if p_counter != 0:
                errmsgs.append(f'SyntaxError: Stray parenthesis')
                flag = False
            return flag
        ##################################################
        
        prev = curr
        # Runs if more than one token left
        while len(q) > 0:
            curr: str = q.popleft()
            
            if curr == '(':
                p_counter += 1
                p_list.append(p_counter)
            
            if curr == ')':
                p_counter -= 1
                p_list.append(p_counter)
            
            # If curr is operator
            if curr in self.operators:
                # Two succeeding operators
                if prev in self.operators:
                    errmsgs.append(f'SyntaxError: Two succeeding operators "{prev} {curr}"')
                    flag = False
                    
            prev = curr
        
        # Should only be one token left when interpreter is here
        
        # If ending with an operator
        if prev in self.operators:
            errmsgs.append(f'SyntaxError: Last token "{prev}" is an operator')
            flag = False
        
        ###### Check paranthesis' #####
        
        # If unbalanced number of parenthesis'
        if p_counter > 0:
            errmsgs.append(f'SyntaxError: Found stray opening parenthesis')
            flag = False
            
        # Check if any negative values in p_list, implies stray closing brackets
        if any((x < 0 for x in p_list)): 
            errmsgs.append(f'SyntaxError: Found stray closing parenthesis')
            flag = False
        ###############################
        return flag

    @staticmethod
    def _get_in_parenthesis(q: deque):
        '''
        Expects proper query
        
        input should be like:
        white AND (woman NOT man))
        i.e. it should miss opening parenthesis
        
        returns deque
        white AND (woman NOT man)
        '''
        q_temp = deque()
        i = 1
        while i > 0:
            token = q.popleft()
            if token == '(':
                i += 1
            elif token == ')':
                i -= 1
                if i == 0: return q_temp
                if i < 0: raise ValueError('Bad query')
            q_temp.append(token)

    def parenthesis_handler(self, querytokens: deque):
        '''
        Expects proper query
        
        Turns deque
        retarded OR (white AND (woman NOT man)) 
        
        to
        ['retarded', 'OR', ['white', 'AND', ['woman','NOT','man']]]
        '''
        q1 = querytokens.copy()
        q2 = deque()
        while len(q1) > 0:
            token = q1.popleft()
            if token == '(':
                q_temp: deque = self._get_in_parenthesis(q1)
                q2.append(self.parenthesis_handler(q_temp))
            else:
                q2.append(token)
        return q2

    # TODO: Think about edge cases 
    # TODO: Maybe use a explicit stack implementation instead of implicit 
    #       to make operator preceedence more modifiable.
    def search(self, query):
        # Obtain query tokens
        querytokens: deque = self.get_logical_querytokens(query)

        # Assert that query is well formatted
        errmsgs = []
        if not self.assert_query(querytokens, errmsgs):
            return None, errmsgs

        # Handle parenthesis 
        queryqueue = self.parenthesis_handler(querytokens)
  
        # Parse queryqueue
        self.temp = set() # Needed to reset TF-IDF values later
        results = self.parse(queryqueue)

        results_list = list(results)
        results_list.sort(key=lambda x: x._tf_idf_score, reverse=True)
        
        scores = np.array([result._tf_idf_score for result in results_list])
        scores = scores/scores.sum()*100

        # Reset _tf_idf_score counter on things in index
        for result in self.temp:
            result._tf_idf_score = 0

        return results_list, scores
    
    def parse(self, q: deque) -> set:
        if len(q) == 0:
            return set()
        results = self._parse_not(q)
        return results

    def _parse_not(self, q: deque):
        results = self._parse_and(q)
        
        if len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == self.difference_operator:
            q.popleft()
            return results - self._parse_not(q)
        else:
            return results

    def _parse_and(self, q: deque) -> Union[set, None]:
        results = self._parse_or(q)
        
        if len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == self.and_operator:
            q.popleft()
            return results & self._parse_and(q)
        else:
            return results

    def _parse_or(self, q: deque) -> Union[set, None]:
        results = self._parse_term(q)
    
        if len(q) == 0:
            return results
        
        curr_token = q[0]
        if curr_token == self.or_operator:
            q.popleft()
            return results | self._parse_or(q)
        else:
            return results

    def _parse_term(self, q: deque) -> Union[set, None]:
        curr_token = q.popleft()

        if type(curr_token) == deque:
            return self.parse(curr_token)

        if curr_token in self.token_to_set:
            results: set = self.token_to_set[curr_token]
            self.temp.update(results)
            self._tf_idf(results, curr_token)
            return results
        else:
            return set()

    def _tf_idf(self, docs: set, token: str):
        '''
        Given docs, calculate TF-IDF
        '''
        for doc in docs:
            idf = np.log(self.n_valid_tokens / len(docs))
            tf = doc.wordfreqs[token] / len(doc)
            doc._tf_idf_score += tf*idf

class Index:
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

    def get_doc(self, uid: str):
        '''
        Get document given uid
        '''
        return self.uid_docmap[uid]

    def _tf_idf(self, result: set, token: str):
        '''
        Given results, calculate TF-IDF
        '''
        for doc in result:
            idf = np.log(len(self.docmap) / len(result))
            tf = doc.wordfreqs[token] / len(doc)
            doc._tf_idf_score += tf*idf

    def search(self, query: Union[str, list], verbose=False) -> tuple:
        '''
        Returns a list of query results given
        query as string or list of strings, also returns tf-idf scores
        '''
        if type(query) == str:
            querytokens: list = clean_text(query)
        elif type(query) == list:
            querytokens: list = query
        else:
            raise ValueError(f'query must be of type string or list, got {type(query)}')

        if len(querytokens) == 0:
            return [], []

        results = set()

        if verbose: print('Query tokens: ', querytokens)
        init_token = querytokens.pop(0)
        if init_token in self.docmap:
            results |= self.docmap[init_token]

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
    def __init__(self, most_similar: Callable, n_similars: int=3):
        super().__init__()
        self.most_similar = most_similar
        self.n_similars = n_similars
    
    def _append_most_similar_tokens(self, tokens: list):
        '''
        Given a list of tokens, for each token in tokens, append the 
        most similar 
        '''
        for token in reversed(tokens):
            similars = [word for word, _ in \
                self.most_similar(token)[:self.n_similars]]
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
    