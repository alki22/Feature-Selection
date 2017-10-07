# Feature selection: Informe
----------------------------------------
## Introducción
--------------
Para este trabajo la tarea fue evaluar los resultados de clustering en vectores sobre los cuales se aplicaron algoritmos de feature selection, tanto supervisado como no supervisado, mediante los cuales se reduce considerablemente la dimensionalidad de los datos a procesar y comparando sus resultados con los del trabajo anterior

## Recursos utilizados
----------------------------------------
### Corpus
Para entrenar el modelo de feature selection supervisado se utilizó el WikiCorpus, el cual fue separado en sucesivas muestras aleatorias de tres documentos cada una para reducir la longitud del mismo y poder ser procesado en su totalidad. En tanto que para el no supervisado y en la parte de clustering se utilizó el corpus compuesto por noticias extraídas de la página web del diario La Voz del Interior provisto por la cátedra.
### Scripts
Fueron realizados mediante el lenguaje de programación Python 3 y las siguientes bibliotecas de código:
  - [__Sci-kit Learn__](http://scikit-learn.org) versión 0.19.0 para las tareas referidas a vectorización y aprendizaje automático.
  - [__Spacy__](http://scikit-learn.org) versión 1.7.0 para el preprocesamiento y tokenización del texto.
  - [__NLTK__](http://www.nltk.org) versión 3.2.5 por su diccionario de stop words, que dio mejores resultados que Spacy.
  - Bibliotecas internas de [__Python 3__](https://docs.python.org/3/)

## Procedimiento
----------------------------------------
### Tokenización
Para tokenizar se usaron estos procedimientos:
1) Para el modelo __no supervisado__ se aplicó el tokenizador incluído en spacy. 
2) Para el modelo __supervisado__ se implementó un tokenizador que separa a cada documento de la muestra aleatoria del corpus en oraciones, identificando a la separación de estas como un doble salto de linea (`text.split('\n\n')`) y, a su vez, estas oraciones en lineas que conformaban tokens con las palabras y su información (palabra-lema-pos-synset)
### Features y vectorización
Para formar los vectores se seleccionaron las siguientes features para cada palabra:
1) Para el modelo __no supervisado__:
    - El lema de la palabra
    - La estimación de su probabilidad logarítmica
    - Su etiqueta POS
    - La tripla de dependencia, conformada por función, palabra objetivo y núcleo
    - El string de la palabra
2) Para el modelo __supervisado__:
    - El string de la palabra
    - El lema de la palabra
    - El synset de WordNet para dicha palabra
    Además se genera un vector unidimensional conteniendo la etiqueta pos de cada token.
    
Luego, se generaron los vectores utilizando [__DictVectorizer__](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html) de Sci-kit Learn
### Feature selection
En ambos casos se optó por reducir el número de features a 350000, ya que para la media de las muestras de tres documentos al azar del WikiCorpus esto representa alrededor del 70% de la información de la palabra (~600000 features inicialmente).
1) Para el modelo __no supervisado__ se aplicó [__TruncatedSVD/LSA__](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) .
2) Para el modelo __supervisado__ se aplicó [__SelectKBest__](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html) usando a [__chi2__](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) como función de puntaje para cada feature, puesto que esta es recomendable en tareas de clasificación. Por último como input además del conjunto de vectores X para las features de las palabras se usó como __target__ al vector unidimensional Y compuesto por las etiquetas pos de cada token.  
### K-Means
Se usó la implementación de Sci-kit Learn del algoritmo sobre la matriz obtenida en el paso anterior con los siguientes parámetros:
- Se estableció en cien el número de clusters.
- Como método de inicialización se utilizó "k-means++", que, según su descripción, selecciona los centroides iniciales para los clusters de forma tal que agiliza la convergencia.

## Evaluación de los resultados
----------------------------------------
### Descargo
Para llegar a evaluar el clustering sobre el corpus, primero se realizó sobre un cuarto y luego la mitad de éste, obteniendo luego resultados similares en la totalidad del corpus. Debe aclararse que la evaluación se hizo "a ojo", por lo que los patrones descriptos en los clusters son puramente una opinión personal. Además, siendo que en el trabajo anterior los resultados fueron evaluados en base a diez clusters, se repitió el experimento con cien clusters para comparar con los resultados presentados a continuación.
### Resultados
En líneas generales, los resultados obtenidos para ambos métodos de feature selection fueron similares a los del trabajo anterior, ya que en ambos se puede notar clusters compuestos por palabras de grupos semánticos similares, por ejemplo clusters con meses, pueblos de Córdoba, movimientos políticos y apellidos de políticos. Sin embargo, cabe destacar que en el modelo de feature selection __supervisado__, a mi parecer por la menor cantidad de features iniciales y por el uso de las etiquetas pos como _target_ del aprendizaje, se aprecia mayor _ruido_ en los clusters y en algunos clusters más correspondencia sintáctica que semántica. Esto se ve reflejado por ejemplo  en clusters compuestos de puros verbos en tiempo pasado y adverbios terminados en _-mente_. 

## Conclusión
----------------------------------------
A la hora de trabajar con representaciones en espacios vectoriales, es importante plantearse la dimensión de nuestro objeto de aprendizaje. Y es a través de mecanismos de __Feature Selection__ que podemos solucionar problemas de escalabilidad y rendimiento de nuestros sistemas para dichas tareas, en algunos casos perdiendo una cantidad poco significante de información pero reduciendo nuestro número de features, y en otros incluso disminuyendo el _ruido_ en nuestras representaciones vectoriales. 


