# -*- coding: utf-8 -*-
"""peer-assesment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1trUSZW77sOmbQSR26ep_laUy84Cv8s_N

<center><img src="https://images.twinkl.co.uk/tw1n/image/private/t_630/image_repo/b8/06/t-lf-242-pupil-voice-learning-child-led-learning-peer-assessment-cards_ver_1.jpg" height="100"></center>

# Detección de incoherencias en evaluación por pares

Profesor: Juan Ramón Rico (<juanramonrico@ua.es>)

## Descripción
---
No cabe duda que las redes neuronales han avanzado en tareas donde se usa texto y valores numéricos. Concretametne en las encuestas o en la evaluación por pares es habitual encontrar valores numéricos (libres o sujetos a una escala - Likert) y comentarios de los evaluadores respecto a las secciones evaluadas en forma de texto (feedback).

- Artículo sobre evaluación por pares donde se han usado redes recurrentes entre otras metodologías para detectar incongruencias en las respuestas <https://www.sciencedirect.com/science/article/pii/S0360131519301629?dgcid=author> ha servido como base para esta actividad.
---

# Introducción

El uso de la evaluación por pares para actividades abiertas tiene ventajas tanto para los profesores como para los estudiantes. Los profesores pueden reducir la carga de trabajo del proceso de corrección y los estudiantes logran una mejor comprensión de la materia al evaluar las actividades de sus compañeros. Para facilitar el proceso, es aconsejable proporcionar a los estudiantes una rúbrica sobre la cual realizar la evaluación de sus compañeros; sin embargo, limitarse a proporcionar sólo puntuaciones numéricas es perjudicial, ya que impide proporcionar una retroalimentación valiosa a otros compañeros. Dado que esta evaluación produce dos modalidades de la misma evaluación, a saber, la puntuación numérica y la retroalimentación textual, es posible aplicar técnicas automáticas para detectar inconsistencias en la evaluación, minimizando así la carga de trabajo de los profesores para supervisar todo el proceso.

Esta actividad estará enfocada en solo una parte de la detección de incongruencias que será la predicción de calificación de una sección usando únicamente información textual.

# Conjunto de datos

Los datos que vamos a utilizar para esta se pueden descargar en un archivo tipo CSV donde figura:

- `activity`: es la actividad desarrollada, en este caso 1 o 2;
-  `year`:  año de comienzo del curso académico estudiado;
- `group`: con valores de 1 si es de mañana o 2 si es de tarde;
- `evaluator`: identificador interno del evaluador para una actividad concreta (la actividad 1 o la 2, de forma excluyente); 
- `work` : es el identificador del trabajo. La entrega se realizaba mediante una URL por lo que hay ocasiones en la que es privada o no accesible y no se ha podido evaluar.
- `secction`: número de sección que se evalua dentro de cada actividad, o 'grade1', 'grade2' cuando se trata de la nota final del trabajo evaluado.
- `value`: valor numérico comprendido entre 0 y 3. Siendo 0 no realizado o realizado incorrectamente y 3 realizado correctamente. Este número puede tener decimales debido a que corresponde al promedio de los valores de la sección. 
- `feedback`: texto libre correspondiente a las recomendaciones que el evaluador realiza en cada sección. Puede estar en blanco lo que indica que no se realizan comentarios y corresponde a que todo está correcto.

Para esta actividad necesitaremos `activity`, `value` y `feeback`.
"""

import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('https://www.dlsi.ua.es/~juanra/UA/dataset/dcadep/dcadep_melt_grades.csv.gz', sep='\t', decimal=',')
data.fillna('', inplace=True) # Reemplazar los valores en blanco por cadena vacías
data = data.sample(frac=1, random_state=123)
display(data.head())
data.dtypes

"""El atributo `section` contiene los identificadores de las diferentes secciones, así como las calificación final (`grade1` y `grade2` cuyos valores están entre 0 y 10) según la actividad. Esta actividad se tiene que evaluar por separado para la actividad 1 o 2 usando únicamente las secciones numéricas (1,2,3,4,5,6 y 7) cuyos valores oscilan entre 0 y 3.

# Visualizar las valoraciones filtrando por secciones correctas (1,2,3,4,5,6,7)
"""

data[data['section'].isin(['1','2','3','4','5','6','7'])][['activity','value']].groupby('activity').hist()

"""Podemos ver como la mayoría de valores son cercanos a 3. No obstante tenemos que predecir cuando este valor va disminuyendo atendiendo a las palabras usadas en el contexto restringido de cada actividad.

##Funciones auxiliares

Nos creamos una funcion auxiliar para imprimir las curvas de aprendizaje de los diferentes modelos
"""

def plot_learning_curves(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Curvas de aprendizaje')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
  plt.show()

"""# Preprocesar el texto

Es necesario preprocesar el texto para descartar símbolos de puntuación, valorar igualmente a palabras en mayúscula y minúscula y extraeer la raiz de las palabras (lemas) para procesarlas como iguales.

Para ello usaremos bibliotecas de procesamiento de lenguaje natural (PLN).
"""

import nltk
nltk.download('stopwords')

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

stemmer = SnowballStemmer("spanish",ignore_stopwords=True)

preprocessed_feedback = []
sentences = []
for i in data.feedback:
  tokens = [stemmer.stem(word) for word in tokenizer.tokenize(i.lower())]
  preprocessed_feedback.append(np.array(' '.join(tokens)))
  sentences.append(' '.join(tokens))

data['feedback prep'] = preprocessed_feedback
data['feedback prep'] = data['feedback prep'].astype('str')
data.head()

"""# Convertir datos de entrenamiento a la forma correcta

En este caso los datos de tipo texto hay transformarlos en secuencias de número que recibirá la red neuronal. 

- En el siguiente enlace <https://www.programcreek.com/python/example/106871/keras.preprocessing.text.Tokenizer> hay varios ejemplos de como convertir el texto que ya tenemos preprocesado a secuencias de números;
- De la secuencia de números hay que aplicar `pad_sequences` como truco para igualar la longitud de todas las secuencias de entreada. Nos facilita la tarea de entrenar una red recurrente con `Keras`.

Seleccionamos las actividades por separado, por lo que vamos a comenzar con la 1.
"""

from tensorflow.keras import preprocessing

# Seleccionar la actividad 1 y sus secciones, ya que se evaluan por separado. La actividad 2 se seleccionaría por separado
activity = 1
new_data = data[(data['activity']==activity) & data['section'].isin(['1','2','3','4','5','6','7'])]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(new_data['feedback prep'])
max_features = 1 + len(tokenizer.word_index)
maxlen = 80

X = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(new_data['feedback prep']), maxlen) # Valores de new_data['feedback prep'] con el padding aplicado

y = new_data['value'].values

embedding_dims = 10

"""Para saber el número de palabras que contienen los textos de feedback preprocesado lo podemos visualizar con (en el ejemplo anterior hemos escogido 80):"""

data['feedback prep num words'] = data['feedback prep'].str.split().apply(len)
data['feedback prep num words'].hist()

"""# Creando la red recurrente tipo LSTM o GRU

Se han preprocesado los datos y los hemos convertido al formato correcto. Ahora tenemos que diseñar nuestra red para entrenarla y realizar las predicciones.

En el caso de las redes recurrentes necesitamos que la entrada cumpla con estas tres dimensiones `(batch_size, time_steps, seq_len)`.

Consideramos la posibilidad de utilizar las versiones de LSTM y GRU optimizadas para CUDA (CuDNNLSTM y CuDNNGRU), ya que funcionará más rápido.
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential  
from tensorflow.keras import layers
import tensorflow as tf

"""# Modelo 1"""

def cnn_model1(input_shape):
  model = keras.Input(shape=input_shape)
  x = layers.Embedding(max_features, embedding_dims, input_length=maxlen)(model)
  x = layers.Dropout(0.2)(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(units=256, return_sequences=True)(x)
  x = tf.compat.v1.keras.layers.CuDNNLSTM(units=128)(x)
  x = layers.Dense(256)(x)
  x = layers.Dense(1)(x)
  outputs = layers.Dense(1, activation='linear')(x)
  model = keras.Model(model, outputs, name='model')
  model.compile(optimizer = 'RMSprop', loss = 'mean_absolute_error', metrics=['mean_squared_error'])

  return model

"""#Tercer modelo"""

def cnn_model3(input_shape):
  
  model3 = keras.Input(shape=input_shape)
  x = layers.Embedding(max_features, embedding_dims, input_length=maxlen)(model3)
  x = layers.Dropout(0.2)(x)
  x = tf.compat.v1.keras.layers.CuDNNGRU(units=256, return_sequences=True)(x)
  x = tf.compat.v1.keras.layers.CuDNNGRU(units=128)(x)
  x = layers.Dense(256)(x)
  x = layers.Dense(1)(x)
  outputs = layers.Dense(1, activation='linear')(x)
  model3 = keras.Model(model3, outputs, name='model3')

  model3.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics=['mean_squared_error']) 

  return model3

"""# Entrenamiento del modelo

Vamos a dividir el conjunto en entrenamiento y test para comprobar nuestro modelo. Para ello tenemos que elegir un tipo de actividad (1 o 2), sus valores y feedback (preprocesado).
"""

from sklearn.model_selection import KFold
n = int(len(new_data)*0.9)


#Resultados del primer clasificador
cvscores = []
cv = KFold(n_splits = 10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  input_shape = X_train.shape[1:]
  model = cnn_model1(input_shape)
  
  
  history = model.fit(X_train, y_train, epochs = 5, validation_split=0.1, batch_size = 32)
  scores = model.evaluate(X_test, y_test, verbose=1)

  print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
  cvscores.append(scores[1] * 100)


# Para verificar que el modelo entrena correctamente crearemos un conjunto de validación del 10% de los datos con el parámetro ()
print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)


print("%.2f%% (+/-%.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
ResultadosPrimerClasificador = cvscores



y_pred = np.clip(model.predict(X_test), 0, 3)
print(f'test mae {sklearn.metrics.mean_absolute_error(y_test, y_pred):.4f}')

"""# Predicciones con el test

Las predicciones  se realizan entre 0 y 3 que son los valores mínimos y máximo establecidos para cada valorar cada ítem.
"""

y_pred = np.clip(model.predict(X_test), 0, 3)
print(f'test mae {sklearn.metrics.mean_absolute_error(y_test, y_pred):.4f}')

"""# Visualización de resultados

Tenemos que crear un DataFrame con el texto del feedback, valores reales, los predichos y su diferencia para apreciar las diferencias.
"""

X_test_new = new_data[len(X_train):][['feedback','value']].copy()
X_test_new['value pred'] = y_pred.ravel()
X_test_new.round(1)

"""# BAG OF WORDS

"""

from tensorflow import keras
documents = []
nltk.download('wordnet')
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stemmer = WordNetLemmatizer()

for sen in range(0, len(data.feedback)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(data.feedback[sen]))
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('spanish'))
X = tfidfconverter.fit_transform(documents).toarray()

from sklearn.model_selection import train_test_split


X_train, X_test = X[:n], X[n:]
y_train, y_test = y[:n], y[n:]
input_shape = X.shape[1:]


model2 = keras.Input(shape=input_shape)
x = layers.Embedding(max_features, embedding_dims, input_length=maxlen)(model2)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(units=64, return_sequences=True)(x)
x = layers.LSTM(units=64)(x)
outputs = layers.Dense(1, activation='linear')(x)
model2 = keras.Model(model2, outputs, name='model')

model2.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics=['mean_squared_error']) 
history = model2.fit(X_train, y_train, epochs = 2, validation_split=0.1, batch_size = 32)

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)

"""#Predicciones

"""

#y_pred = model.predict(X_test)


#print(f'test mae {sklearn.metrics.mean_absolute_error(y_test, y_pred):.4f}')

"""# Visualizaciones de resultados"""

#X_test_new = new_data[n:][['feedback','value']].copy()
#print(np.size(y_pred))
#X_test_new['value pred'] = y_pred.ravel()
#X_test_new.round(1)

"""# Evaluación de redes por actividad"""

from tensorflow.keras import preprocessing


# Seleccionar la actividad 1 y sus secciones, ya que se evaluan por separado. La actividad 2 se seleccionaría por separado
activity = 2
new_data = data[(data['activity']==activity) & data['section'].isin(['1','2','3','4','5','6','7'])]
tokenizer = preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(new_data['feedback prep'])
max_features = 1 + len(tokenizer.word_index)
maxlen = 80

X = preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences(new_data['feedback prep']), maxlen) # Valores de new_data['feedback prep'] con el padding aplicado

y = new_data['value'].values

embedding_dims = 10

"""# Modelo para actividad 2

"""

input_shape = X.shape[1:]
model = keras.Input(shape=input_shape)
x = layers.Embedding(max_features, embedding_dims, input_length=maxlen)(model)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(units=64, return_sequences=True)(x)
x = layers.LSTM(units=64)(x)
outputs = layers.Dense(1, activation='linear')(x)
model = keras.Model(model, outputs, name='model')



model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics=['mean_squared_error'])

from sklearn.model_selection import KFold
n = int(len(new_data)*0.9)

cv = KFold(n_splits = 10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

# Para verificar que el modelo entrena correctamente crearemos un conjunto de validación del 10% de los datos con el parámetro ()
history = model.fit(X_train, y_train, epochs = 5, validation_split=0.1, batch_size = 32)

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)

y_pred = np.clip(model.predict(X_test), 0, 3)
print(f'test mae {sklearn.metrics.mean_absolute_error(y_test, y_pred):.4f}')

X_test_new = new_data[12046:][['feedback','value']].copy()
print(np.size(y_pred))
X_test_new['value pred'] = y_pred.ravel()
X_test_new.round(1)

"""# WILCOXON RANKED TEST
##  Tercer modelo
"""

import tensorflow as tf



#Resultados del tercer clasificador
cvscores3 = []

cv = KFold(n_splits = 10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
    
  input_shape = X.shape[1:]

  model3 = cnn_model3(input_shape)
  history = model3.fit(X_train, y_train, epochs = 5, validation_split=0.1, batch_size = 32)
  scores3 = model3.evaluate(X_test, y_test, verbose=1)

  print("%s: %.2f%%" % (model3.metrics_names[1], scores3[1] * 100))
  cvscores3.append(scores3[1] * 100)

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)

print("%.2f%% (+/-%.2f%%)" % (np.mean(cvscores3), np.std(cvscores3)))
ResultadosTercerClasificador = cvscores3

print("%.2f%% (+/-%.2f%%)" % (np.mean(cvscores3), np.std(cvscores3)))
ResultadosTercerClasificador = cvscores3
y_pred = np.clip(model3.predict(X_test), 0, 3)
print(f'test mae {sklearn.metrics.mean_absolute_error(y_test, y_pred):.4f}')

from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')


wilcox_V, p_value =  wilcoxon(ResultadosPrimerClasificador, ResultadosTercerClasificador, alternative='greater', zero_method='wilcox', correction=False)

print('Resultado completo del test de Wilcoxon')
print(f'Wilcox V: {wilcox_V}, p-value: {p_value:.2f}')

"""# SHAPLEY"""

!pip install shap 
import xgboost
import shap

# load JS visualization code to notebook
shap.initjs()


model4 = xgboost.XGBRegressor(3, 0.1, 100, 1)
model4.fit(X=X_train, y=y_train)
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.TreeExplainer(model4)
shap_values = explainer.shap_values(X)

print(X_train.shape)
X_train_df = pd.DataFrame(data=X_train)
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train_df.iloc[0,:])