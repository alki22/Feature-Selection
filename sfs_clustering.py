from __future__ import print_function
import sys
import nltk
import spacy
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def get_token_features(token):
    return {
            'lemma' : token.lemma, 'shape' : token.shape,
            'log probability' : token.prob,'string' : token.orth_,
            'dependency' : token.dep, 'head' : token.head.text,
            'name' : token.orth, 'tag' : token.tag,
            'prefix' : token.prefix, 'suffix' : token.suffix
            }

file_name = "lavoztextodump.txt"
text = open(file_name, 'r').read()

parser = spacy.load('es')
parsed_data = parser(text)

stopwords = nltk.corpus.stopwords.words('spanish')

corpus = []
pos_tags = []
vect_to_string = []

for token in parsed_data:
    # Clean the data out of stopwords and non alphabetic tokens
   if token.orth_.isalpha() and token.orth_.lower() not in stopwords: 
        if not any(features['string'] == token.orth_ for features in corpus):
            corpus.append(get_token_features(token))
            pos_tags.append(token.pos_)
            vect_to_string.append(token.orth_)

vectorizer = DictVectorizer()
vectors = vectorizer.fit_transform(corpus)

# Load selector model
selector = joblib.load('selector_sup.pkl')
reduced_vectors = selector.fit_transform(vectors, pos_tags)

# Run K-Means algorithm
k_clusters = 100
km = KMeans(n_clusters=k_clusters, init='k-means++', n_jobs=-1)
X = km.fit(reduced_vectors)

# Put every word's index into its cluster
clusters = defaultdict(list)
for j in range(len(vect_to_string)):
    clusters[X.labels_[j]].append(j)

# Print results to output.txt
with open("output.txt", "a") as f:
    for j in range(len(clusters)):
        print('----------------------------------', file=f)
        print('Cluster', j, file=f)
        for k in clusters[j]:
                print(vect_to_string[k], file=f)

joblib.dump(km,  'clasificador_s.pkl')