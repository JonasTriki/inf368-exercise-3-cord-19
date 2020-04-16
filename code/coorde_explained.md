# Creating a search engine using Word2Vec and TD-IDF

## Motivation and concept
We want to obtain answers to the problems by creating our own search engine for the dataset. Naturally, we want the search engine to be fast and give relevant results when given a query. 

To achieve the desired searching speeds we pre-compute an index for the data. The index is a [hash table](https://en.wikipedia.org/wiki/Hash_table) that maps query tokens (e.g. words) to [sets](https://en.wikipedia.org/wiki/Set_(mathematics)) of documents that contains the tokens. 

To achieve relevant search results, we analyze the query and add tokens that are similar by utilizing word2vec. For example given the query "fat cat", we may effectively turn the query into "fat obese cat dog". The sets corresponding to each query can then for example be combined to a larger set. Then to score the relevance of the queried documents, we calculate the [TF-IDF](http://www.tfidf.com/) weights for each document. 

For example, when given the query "covid symptoms" the search engine essentially does:
1.  Insert similar tokens to the query, e.g. corona and diagnose, effectively turning the query into "covid corona symptoms diagnose" 
2.  Obtain set $A$ of documents that contains "covid"
3.  Obtain set $B$ of documents that contains "corona"
4.  Obtain set $C$ of documents that contains "symptoms"
5.  Obtain set $D$ of documents that contains "diagnose"
6.  Obtain the union $E$ of the sets
7.  Calculate the TF-IDF weights for each document in $E$ with respect to the query tokens
8.  Return the documents sorted by the TF-IDF weights in descending order

### Query syntax with set operators
Sometimes we would like to specify a more refined query. For example what if specifically we want the documents that contains the words "corona" and "influenza", without the word "swine". To enable this ability to the user, the search engine implements a parser that can parse set operators as well.

<a id='syntax'></a>
The search engine supports three operators: OR (union), AND (intersection), NOT (difference), from highest to lowest preceedence, meaning that the OR operator is evaluated before AND and so on. To override the order of preceedence, the user can use parenthesis', e.g. "(cat AND dog) OR goose" will be different from "cat AND dog OR goose". To sum it up with another example:

Given the query "cat AND virus" the search engine essentially does:
1.  Insert similar tokens in the query, e.g. dog and disease, effectively turning the query into "(cat OR dog) AND (virus OR disease)" 
2.  Obtain set $A$ set of documents that contains "cat"
3.  Obtain set $B$ set of documents that contains "dog"
4.  Obtain set $C$ set of documents that contains "virus"
5.  Obtain set $D$ set of documents that contains "disease"
6.  Let set $E$ be the result of evaluating $(A \cup B) \cap (C \cup D)$
7.  Calculate the TF-IDF weights for each document in $E$ with respect to the query tokens
8.  Return the documents sorted by the TF-IDF weights in descending order

Practically, the implementation of the search engine automatically adds OR operators between tokens that does not have any explicit operators, e.g. "cat dog horse AND goose" $\Rightarrow$ "cat OR dog OR horse AND goose ". To explicitly illustrate the effect of the preceedence, the query is equivalent to "(cat OR dog OR horse) AND goose".

## Indexing process
We will now give a high level explanation on how the indexing process works. The source code for the indexing process is available [here](https://github.com/JonasTriki/inf368-exercise-3-coordle/blob/master/coordle/backend/coordle_backend.py#L537) if you so desire to examine the implementation details. 

Given a document $d_i$ from the collection of all documents $D$ that has an unique id $u_i$, and its corresponding text $t_i$ we do the following:
1. Clean the text using the function ``clean_text`` mentioned in [section 3](#create_word2vec). The return value is a list containing text tokens. 
2. Get the unique tokens $\tau_{ij}$ and their counts $c_{ij}$. ($i$ denotes document, $j$ denotes token index)
3. For the document, create a dictionary $f_i$ that maps each token $\tau_{ij}$ to their corresponding counts $c_{ij}$. 

We repeat this process for all the documents in $D$. 

## Parser 
Without going into the details; the parsing method used to parse the [syntax](#syntax) is [recursive descent parsing](https://en.wikipedia.org/wiki/Recursive_descent_parser). The source code for the parser part of the source engine can be found [here](https://github.com/JonasTriki/inf368-exercise-3-coordle/blob/master/coordle/backend/coordle_backend.py#L421). 

### Things to note about syntax
As a design choice, when using the NOT operator (set difference), the search engine will not append similar terms to the term that is used for the difference. For example: "cat AND car" may effectively become "(cat OR dog) AND (car OR bus)", while "cat NOT car" becomes "(cat OR dog) NOT car". This is to avoid removing potential relevant results. 

An issue that arises (that is yet to be handled) is that given the query "cat NOT car sheep" (all documents with cats but not car or sheep), the query effectively becomes for example "(cat AND dog) NOT car OR (sheep OR cow)", which effectively is "(cat AND dog) NOT (car OR (sheep OR cow))" because of operator preceedence. This may lead to a removal of a too large subset. For now the workaround the user has to do to ensure "proper behaviour" when using NOT is to chain them together; e.g "cat NOT car NOT sheep".

## TD-IDF 
[TF-IDF](http://tfidf.com/) stands for *term frequency-inverse document frequency*. It is a weight that is used often used for information retrieval and text mining. Given a collection of documents, TF-IDF is a statistical measure used to determine how strongly connected a word is to a specific document relatively to all the other documents. Given a word and a document from the collection, we compute TF (term frequency) and multiply it with IDF (inverse document frequency) to obtain its TF-IDF weight. The calculations are as follows:

TF = (number of times the word appears in the document) / (total number of words in said document)

IDF = log(documents in collection / number of documents with the word in it)

The important parts to note for the query process are:
- Given a query $Q$, the search engine must retrieve sets for each query token $q_i$.
- Query $Q$ can have repeated tokens.
- Right after retrieving the set corresponding to $q_i$, it calculates the TF-IDF score for each document in the set.
- If a document is retrieved multiple times, it will get accumulate TF-IDF score.


  
