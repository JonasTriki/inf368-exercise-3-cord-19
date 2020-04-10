from collections import deque
from typing import Union, Callable

class RecursiveDescentParser:
    '''
    Parser class for Coordle

    Made to parse query tokens contained 
    in deque objects.

    This class is made to be composed in coordle_backend.Index
    '''
    def __init__(self, token_to_set: dict):
        '''
        Parameters:
        ------------
        sets: a dictionary where keys correspond to query tokens, and values
              are sets
        '''
        self.token_to_set = token_to_set

    def _parse_not(self, q: deque):
        results = self._parse_and(q)
        
        # Return if results is None, or if token queue is empty
        if results is None or len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == 'NOT':
            q.popleft()
            return results - self._parse_not(q)
        else:
            return results

    def _parse_and(self, q: deque) -> Union[set, None]:
        results = self._parse_or(q)
        
        # Return if results is None, or if token queue is empty
        if results is None or len(q) == 0:
            return results

        curr_token = q[0]
        if curr_token == 'AND':
            q.popleft()
            return results & self._parse_and(q)
        else:
            return results

    def _parse_or(self, q: deque) -> Union[set, None]:
        results = self._parse_set(q)
    
        # Return if results is None, or if token queue is empty
        if results is None or len(q) == 0:
            return results
        
        curr_token = q[0]
        if curr_token == 'OR':
            q.popleft()
            return results | self._parse_or(q)
        else:
            return results

    def _parse_set(self, q: deque) ->Union[set, None]:
        curr_token = q[0]
        if curr_token in self.token_to_set:
            q.popleft()
            return self.token_to_set[curr_token]
        else:
            return None