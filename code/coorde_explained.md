# Creating a search engine using Word2Vec and TD-IDF

## Motivation and concept
We want to obtain answers to the problems by creating our own search engine for the dataset. Naturally, we want the search engine to be fast and give relevant results when given a query. 

To achieve the desired searching speeds we pre-compute an index for the data. The index is a [hash table](https://en.wikipedia.org/wiki/Hash_table) that maps query tokens (e.g. words) to [sets](https://en.wikipedia.org/wiki/Set_(mathematics)) of documents that contains the tokens. 

To achieve relevant search results, we analyze the query and add tokens that are similar. For example given the query "fat cat", we may effectively turn the query into "fat obese cat dog". Then to score the relevance of the queried documents, we calculate the [TF-IDF](http://www.tfidf.com/) weights for each document. 

For example, when given the query "covid symptoms" the search engine essentially does:
1.  Insert similar tokens to the query, e.g. corona and diagnose, effectively turning the query into "covid corona symptoms diagnose" 
2.  Obtain set $A$ of documents that contains "covid"
3.  Obtain set $B$ of documents that contains "corona"
4.  Obtain set $C$ of documents that contains "symptoms"
5.  Obtain set $D$ of documents that contains "diagnose"
6.  Obtain the union $E$ of the sets
7.  Calculate the TF-IDF weights for each document in $E$
8.  Return the documents sorted by the TF-IDF weights in descending order. 

### Query syntax with set operators
Sometimes we would like to give a more refined query. For example what if specifically we want the documents that contains the words corona and influenza, without the word swine. To enable this ability to the user, the search engine implements a parser that can parse set operators as well, we are working with sets after all. 

The search engine supports three operators: OR (union), AND (intersection), NOT (difference), from highest to lowest preceedence, meaning that the OR operator is evaluated before AND and so on. To override the order of preceedence, the user can use parenthesis', e.g. "(cat AND dog) OR goose" will be different from "cat AND dog OR goose". To sum it up with another example:

Given the query "cat AND virus" the search engine essentially does:
1.  Insert similar tokens in the query, e.g. dog and disease, effectively turning the query into "(cat OR dog) AND (virus OR disease)" 
2.  Obtain set $A$ set of documents that contains "cat"
2.  Obtain set $B$ set of documents that contains "dog"
2.  Obtain set $C$ set of documents that contains "virus"
2.  Obtain set $D$ set of documents that contains "disease"
3.  Let $E$ be the result of evaluating $(A \cup B) \cap (C \cup D)$
4.  Calculate the TF-IDF weights for each document in $E$
5.  Return the documents sorted by the TF-IDF weights in descending order. 

Practically, the implementation of the search engine automatically adds OR operators between tokens that does not have any explicit operators, e.g. "cat dog horse AND goose" $\Rightarrow$ "cat OR dog OR horse AND goose ". To explicitly illustrate the effect of the preceedence, the query is equivalent to "(cat OR dog OR horse) AND goose".

