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
from pymongo import MongoClient, collection

# TODO: Return suggestions so website can highlight

class RecursiveDescentParser:
    '''
    Parser class for Coordle used for parsing queries
    and also searching using the parsed queries.

    Made to parse query tokens contained
    in deque objects.

    This class is made to be composed in coordle_backend.Index
    '''
    def __init__(self, index: 'Index', or_operator: str='OR',
                 and_operator: str='AND', difference_operator: str='NOT',
                 punctuation: str=None):
        '''
        Parameters:
        ------------
        sets: a dictionary where keys correspond to query tokens, and values
              are sets containing CordDoc objects
        '''
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

        self.word2uids = index.word2uids
        self.wordcounts = index.wordcounts
        self.index = index

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
        # Runs if any tokens left in queue
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

        # If ending with an operator
        if prev in self.operators:
            errmsgs.append(f'SyntaxError: Last token "{prev}" is an operator')
            flag = False

        ###### Check paranthesis' #####

        # If unbalanced number of parenthesis'
        if p_counter > 0:
            errmsgs.append(f'SyntaxError: Found stray opening parenthesis')
            flag = False

        # Check if any negative values in p_list, implies stray closing
        # parenthesis
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

    def clean_tokens(self, q: deque, cleaner: Callable = None) -> deque:
        '''
        Given a logical query (that is queries with set operators and
        parenthesis'), run a text cleaning function on the tokens that are not
        operators or parenthesis'
        '''

        if cleaner is None:
            cleaner = clean_text

        q2 = deque()
        for token in q:
            if token in '()' or token in self.operators:
                q2.append(token)
            else:
                q2.append(clean_text(token, return_list=False))
        return q2

    def _preprocess_query(self, query: str) -> tuple:
        '''
        Preprocesses the query before parsing

        Returns queryqueue, a deque that contains strings and deques.
        '''
        querytokens: deque = self.get_logical_querytokens(query)

        # Assert that query is well formatted
        errmsgs = []
        if not self.assert_query(querytokens, errmsgs):
            return None, errmsgs

        # "Preprocess tokens" by cleaning tokens that are terms (words)
        querytokens = self.clean_tokens(querytokens)

        # Handle parenthesis
        queryqueue = self.parenthesis_handler(querytokens)

        return queryqueue, errmsgs

    def __getitem__(self, query: str):
        '''
        Implements fancy syntax: coordle['query here']
        '''
        return self.search(query)

    def search(self, query: Union[str, deque]) -> tuple:
        '''
        TODO: Docs
        '''
        if type(query) == str:
            queryqueue, errmsgs = self._preprocess_query(query)
        elif type(query) == deque:
            queryqueue, errmsgs = query, []
        else:
            raise ValueError(f'Got unsupported type for query, got {type(deque)}')

        # If invalid query
        if queryqueue is None:
            return None, None, errmsgs

        # Parse queryqueue
        self.temp = {}

        results = self.parse(queryqueue)

        results_list = np.array(list(results))
        scores = np.array([self.temp[doc.uid] for doc in results_list])

        sort_idx = np.argsort(scores)[::-1]
        results_list = results_list[sort_idx]
        scores = scores[sort_idx]

        scores = scores/scores.sum()*100

        return results_list, scores, errmsgs

    def parse(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        if len(q) == 0:
            return set()

        results = self._parse_not(q)
        return results

    def _parse_not(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        results = self._parse_and(q)

        if len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == self.difference_operator:
            q.popleft()
            return results - self._parse_not(q)
        else:
            return results

    def _parse_and(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        results = self._parse_or(q)

        if len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == self.and_operator:
            q.popleft()
            return results & self._parse_and(q)
        else:
            return results

    def _parse_or(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        results = self._parse_term(q)

        if len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == self.or_operator:
            q.popleft()
            return results | self._parse_or(q)
        else:
            return results

    def _parse_term(self, q: deque) -> set:
        '''
        TODO: Docs
        '''
        curr_token = q.popleft()

        # Implies parenthesis in query
        if type(curr_token) == deque:
            return self.parse(curr_token)

        if curr_token in self.token_to_set:
            results: set = self.token_to_set[curr_token]
            self._tf_idf(results, curr_token)
            return results
        else:
            return set()

    def _tf_idf(self, docs: set, token: str):
        '''
        TODO: OUTDATED DOCS
        Given a set CordDoc objects and a token, calculate TF-IDF relevance for
        the objects with respect to the token and then store the value inside
        the objects.
        '''
        for uid in docs:
            idf = np.log(len(self.index) / len(docs))
            # self.wordcounts.find_one({'_id':uid})
            tf = doc.wordfreqs[token] / len(doc)
            if uid not in self.temp:
                self.temp[uid] = 0
            self.temp[uid] += tf*idf


class Index:
    '''
    Index object for Cord data
    '''
    def __init__(self, db: str, host=None, port: int=None, 
                 maxPoolSize: int=1000000, wordcounts: str='wordcounts',
                 word2uids: str='word2uids', drop_old_collections: bool=True):
        # self.rdp = RecursiveDescentParser(self.docmap)
        self.mongoclient = MongoClient(host=host, port=port,
                                       maxPoolSize=maxPoolSize)
        self.db = self.mongoclient[db]
        
        if drop_old_collections:
            if wordcounts in self.db.list_collection_names():
                self.db[wordcounts].drop()
            if word2uids in self.db.list_collection_names():
                self.db[word2uids].drop()

        self.wordcounts: collection = self.db[wordcounts]
        self.word2uids: collection = self.db[word2uids]

        self.rdp = RecursiveDescentParser(self)
        self.len = 0

    def __len__(self):
        '''
        Implements polymorphism for len function
        '''
        return self.len

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
        # Make wordfreqs
        doc = CordDoc(uid=uid, title=title)
        doc, unique_tokens = doc.fit(text)

        # Add document to hasmap where keys are uids and values are docs
        self.uid_docmap[doc.uid] = doc
        self.len += 1

        # Add document to hashmap where keys are unique tokens, and values
        # are sets
        for token in unique_tokens:
            if token not in self.docmap:
                self.docmap[token]=set()
            self.docmap[token].add(doc)

    @staticmethod
    def wordcount_generator(uids, texts):
        for uid, text in zip(uids, texts):
            yield {'_id':uid, 
                   'wordcounts':pd.value_counts(text, sort=False).to_dict()}
    
    @staticmethod
    def word2uid_generator(docmap: dict, word_maxlen: int=500):
        for word, uids in docmap.items():
            if len(word) < word_maxlen:
                yield {'_id':word, 'uids':{uid:None for uid in uids}}

    def build_from_df(self, df: pd.DataFrame, uid: str, title: str,
                      text: str, use_multiprocessing: bool=False,
                      workers: int=1, verbose: bool=True,
                      cleaner: Callable=None):
        '''
        Build index given df, a pd.DataFrame.

        Parameters
        -----------
        df: pd.DataFrame at least containing separate columns for uid (unique
            id), title and text.

        uid: name of uid column

        title: name of title column

        text: name of text column

        use_multiprocessing: Optional, if True (False by default) it will
                             preprocess text using threads.

        workers: Optional, specifies number of workers to use if
                 use_multiprocessing is True.
        '''
        if cleaner is None:
            cleaner = clean_text

        tqdm_kwargs = {'position':0, 'disable':not verbose}

        if use_multiprocessing:
            # Clean texts on multiple cores
            if workers == -1:
                workers = cpu_count()

            if verbose:
                print(f'Text cleaning initilized on {workers} workers')

            with Pool(workers) as pool:
                clean_generator = tqdm(
                    pool.imap(cleaner, df[text]),
                    desc='Cleaning texts', total=len(df),
                    **tqdm_kwargs
                )
                texts=list(clean_generator)
        else:
            clean_generator = tqdm(
                (cleaner(t) for t in df[text]),
                desc='Cleaning texts', total=len(df),
                **tqdm_kwargs
            )
            texts = list(clean_generator)

        uids = df[uid]
        titles = df[title]

        # wc_gen = tqdm(self.wordcount_generator(uids, texts), 
        #               desc='Inserting word counts to database', total=len(df), 
        #               **tqdm_kwargs)
        # self.wordcounts.insert_many(wc_gen)

        self.docmap = dict()
        self.len += len(df)

        for uid_, text_ in tqdm(zip(uids, texts), 
                                desc='Building word to uids map',
                                total=len(df), **tqdm_kwargs):
            counts: pd.Series = pd.value_counts(text_, sort=False)
            
            unique_tokens = counts.index.values
            self.wordcounts.insert_one(
                {'_id':uid_, 'wordcounts':counts.to_dict()})

            # Add document to hashmap where keys are unique tokens, and values
            # are sets
            for token in unique_tokens:
                if token not in self.docmap:
                    self.docmap[token]=set()
                self.docmap[token].add(uid_)  

        wm_gen = tqdm(self.word2uid_generator(self.docmap, 100), 
                        total=len(self.docmap), 
                        desc='Inserting wordmap to database', **tqdm_kwargs)
        self.word2uids.insert_many(wm_gen)

    def extend_with_df(self, df: pd.DataFrame, uid: str, title: str,
                       text: str, use_multiprocessing: bool=False,
                       workers: int=1, verbose: bool=True,
                       cleaner: Callable=None):
        '''Heh'''

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
        return self.rdp.search(query)


class AI_Index(Index):
    '''
    Essentially, uses TF-IDF, but adds similar query tokens
    to given query using AI, Big Data and Machine Learning $$$
    '''
    def __init__(self, most_similar: Callable, n_similars: int=3):
        super().__init__()
        self.most_similar = most_similar
        self.n_similars = n_similars

    def _similar_adder(self, q: deque, token: str):
        '''
        Appends similar words to given token to queue
        '''
        try:
            similars = [word for word, _ in \
                        self.most_similar(token)[:self.n_similars]]

            for word in similars:
                q.append(self.rdp.or_operator)
                q.append(word)
        except KeyError:
            pass

    def _add_most_similar_tokens(self, q: deque):
        '''
        Assumes proper query

        Given a list of tokens, for each token in tokens, append the
        most similar
        '''

        q1 = q.copy()
        q2 = deque()

        # Initialize by checking first element
        token = q1.popleft()
        q2.append(token)

        if not (token in '()' or token in self.rdp.operators):
            self._similar_adder(q2, token)

        for token in q1:
            q2.append(token)
            if token in '()' or token in self.rdp.operators:
                continue

            # Dont add extra tokens if NOT operator
            if q2[-2] == self.rdp.difference_operator:
                continue

            self._similar_adder(q2, token)

        return q2

    def _preprocess_query_with_ai(self, query: str) -> tuple:
        '''
        Preprocesses the query before parsing
        '''
        querytokens: deque = self.rdp.get_logical_querytokens(query)

        # Assert that query is well formatted
        errmsgs = []
        if not self.rdp.assert_query(querytokens, errmsgs):
            return None, errmsgs

        # "Preprocess tokens" by cleaning tokens that are terms (words)
        querytokens = self.rdp.clean_tokens(querytokens)

        # This is the "AI" part. This will return querytokens that has gotten
        # appended similar tokens to the already existing tokens
        querytokens = self._add_most_similar_tokens(querytokens)

        # Handle parenthesis
        queryqueue = self.rdp.parenthesis_handler(querytokens)

        return queryqueue, errmsgs

    def search(self, query: str) -> tuple:
        '''
        Returns a list of query results given
        query as string or list of strings, also returns tf-idf scores
        '''
        queryqueue, errmsgs = self._preprocess_query_with_ai(query)

        if queryqueue is None:
            # Should always be error message if anything is wron with query
            assert len(errmsgs) > 0
            return None, None, errmsgs

        return self.rdp.search(queryqueue)





