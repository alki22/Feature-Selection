from __future__ import print_function
import sys
import nltk
import spacy
from sklearn.cluster import KMeans
from collections import defaultdict
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

# (Vector number <-> word) list
vect_to_string = []
print('Creando corpus')
for token in parsed_data:
    # Clean the data out of stopwords and non alphabetic tokens
   if token.orth_.isalpha() and token.orth_.lower() not in stopwords: 
        if not any(features['string'] == token.orth_ for features in corpus):
            corpus.append(get_token_features(token))
            vect_to_string.append(token.orth_)

vectorizer = DictVectorizer()
vectors = vectorizer.fit_transform(corpus)

# Reduce vectors' dimension
#vt = VarianceThreshold(threshold=(.15))
#reduced_vectors = vt.fit_transform(vectors)

# Run K-Means algorithm
k_clusters = 100
km = KMeans(n_clusters=k_clusters, init='k-means++', n_jobs=-1)
X = km.fit(vectors)
joblib.dump(km,  'clasificador.pkl')

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
