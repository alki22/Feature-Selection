import sys
import nltk
import random
from os import listdir
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
document_list = listdir('./tagged.es')

selector = SelectKBest(score_func=chi2, k=350000) 

#cosas que no queremos
stopwords = nltk.corpus.stopwords.words('spanish')
punctuation = ['(', ')', '?', ':', ';', ',', '.', '!', '/', '"', "'"]

#muestra aleatoria del conjunto de documentos
while (document_list):
	sample = document_list
	if(len(document_list) >= 3):
		sample = random.sample(document_list, 3)
	
	corpus = []
	pos_tags = []
	print('Preproceso')	
	for filename in sample:
		path = './tagged.es/' + filename
		file = open(path, 'r')
		text = file.read()
		
		#separar el texto en oraciones
		sentences = text.split('\n\n')
		
		#separar oraciones en lineas
		for sentence in sentences:
			if not sentence.startswith('<'):
				lines = sentence.split('\n')
				
				#convertir lineas en tokens y obtener sus features
				for line in lines:
					token = line.split()
					if (token[0] not in punctuation and token[0] not in stopwords and
						token[1] not in stopwords and len(token) > 3):
						features = {
									'word' : token[0], 
									'lemma' : token[1],
									'synset' : token[3]
									}
						corpus.append(features)
						pos_tags.append(token[2])
	print('Vectorización')
	
	#vectorizar
	vectorizer = DictVectorizer()
	vectors = vectorizer.fit_transform(corpus)	

	print('Reducción de dimensionalidad')
	
	#reducir dimensionalidad
	reduced_vectors = selector.fit_transform(vectors, pos_tags)
	
	#excluir la muestra de los documentos restantes
	document_list = list(set(document_list)^set(sample))

print('Guardado de modelo')
joblib.dump(selector,'selector_sup.pkl')