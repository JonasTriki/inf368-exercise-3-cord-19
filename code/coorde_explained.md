# Creating a search engine using Word2Vec and TD-IDF

## Motivation and concept
We want to obtain answers to the problems by creating our own search engine for the dataset. Naturally, we want the search to be fast and give relevant results when given a query. 

To achieve the desired searching speeds we pre-compute an index for the data. The index is a [hash table](https://en.wikipedia.org/wiki/Hash_table) that maps query tokens (e.g. words) to [sets](https://en.wikipedia.org/wiki/Set_(mathematics)) of documents that contains the tokens. 

To achieve relevant search results, we analyze the query and add tokens that are similar. For example given the query "fat cat", we effectively turn the query into "fat obese cat dog". Then to score the relevance of the queried documents, we calculate the [TF-IDF](http://www.tfidf.com/) weights for each document. 

For example, when given the query "covid symptoms" the search engine does:
1.  Insert similar tokens to the query, e.g. corona and diagnose, effectively turning the query into "covid corona symptoms diagnose" 
2.  Obtain set $A$ of documents that contains "covid"
3.  Obtain set $B$ of documents that contains "corona"
4.  Obtain set $C$ of documents that contains "symptoms"
5.  Obtain set $D$ of documents that contains "diagnose"
6.  Obtain the union $E$ of the sets
7.  Calculate the TF-IDF weights for each document in $C$
8.  Return the documents sorted by the TF-IDF weights in descending order. 

### Query syntax with set operators
Sometimes we would like to give a more refined query. For example what if specifically we want the documents that contains the words corona and influenza, without the word swine. To enable this ability to the user, the search engine implements a parser that can parse set operators as well, we are working with sets after all. 

The search engine supports three operators: OR (union), AND (intersection), NOT (difference), order respectively from highest to lowest preceedence, meaning that the OR operator is evaluated before AND and so on. To override the order of preceedence, the user can use parenthesis', e.g. "(cat AND dog) OR goose" will be different from "cat AND dog OR goose". 