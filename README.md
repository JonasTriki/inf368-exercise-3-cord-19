# INF368 Exercise 3 Spring 2020
In the third exercise for the Selected Topics in Machine Learning (topic being "Deep Learning") course ([INF368 Spring 2020](https://www.uib.no/en/course/INF368?sem=2020v)) at the University of Bergen, we are asked to participate in the [COVID-19 Open Research Dataset Challenge (CORD-19)](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).

## Kaggle challenge
We decided to create a search engine for the articles in the CORD-19 dataset using TF-IDF and word embeddings using Word2Vec.

#### TF-IDF 
[TF-IDF](http://tfidf.com/) stands for *term frequency-inverse document frequency*. It is a weight that is used often used for information retrieval and text mining. Given a collection of documents, TF-IDF is a statistical measure used to determine how strongly connected a word is to a specific document relatively to all the other documents. Given a word and a document from the collection, we compute TF (term frequency) and multiply it with IDF (inverse document frequency) to obtain its TF-IDF weight. The calculations are as follows:

TF = (number of times the word appears in the document) / (total number of words in said document)

IDF = log(documents in collection / number of documents with the word in it)

### Word2Vec
We use word embeddings to enhance the search by appending similar words. For instance, if we search for the word "covid-19", the search engine will also use similar terms such as "coronavirus" or "sars". The motivation to create such a search engine was to learn how TF-IDF and word embeddings work in a more practical setting. We use [Gensims implementation of Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) to create our word embeddings using the negative sampling technique, as described in the following paper: [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf).

## Kaggle tasks
In the [Kaggle notebook](https://www.kaggle.com/trikiuib/coordle-search-engine-using-word2vec-and-tf-idf), we go into more depth of how our system works. In particular, we answer the following tasks:
* What do we know about COVID-19 risk factors?
* What do we know about vaccines and therapeutics?
* What has been published about medical care?
* What do we know about diagnostics and surveillance?

## Authors
The main authors of this repository are [Jonas Triki](https://github.com/JonasTriki) and [Naphat Amundsen](https://github.com/Laohachot). This repository will contain the main code for our submission to the Kaggle competition.
