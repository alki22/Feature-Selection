from __future__ import print_function
import sys
import nltk
import spacy
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer

def get_token_features(token):
    return {
            'lemma' : token.lemma, 'shape' : token.shape,
            'log probability' : token.prob, 'POS tag' : token.pos,
            'dependency' : token.dep, 'head' : token.head.text,
            'name' : token.orth, 'tag' : token.tag,
            'prefix' : token.prefix, 'suffix' : token.suffix,
            'string' : token.orth_
            }

parser = spacy.load('es')

# dataset/file to cluster
file_name = "lavoztextodump.txt"
text = open(file_name, 'r').read()

parsed_data = parser(text)
stopwords = nltk.corpus.stopwords.words('spanish')

corpus = []

for token in parsed_data:
    # Clean the data out of stopwords and non alphabetic tokens
   if token.orth_.isalpha() and token.orth_.lower() not in stopwords: 
        if not any(features['string'] == token.orth_ for features in corpus):
            corpus.append(get_token_features(token))

vectorizer = DictVectorizer()
vectors = vectorizer.fit_transform(corpus)

selector = TruncatedSVD(n_components=1000)
reduced_vectors = selector.fit_transform(vectors)

joblib.dump(selector,'selector_unsup.pkl')
