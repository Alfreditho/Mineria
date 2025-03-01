import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re


nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

mensajes = [
    "En metro ayer nos atendieron bien y los desayunos son una delicia",
    "Los mejores desayunos... Y sin duda el mejor cafe, mi récord personal son 5 tazas en un desayuno!!!",
    "El servicio al cliente es excelente",
    "En el puerto de la libertad dan un pesimo servicio",
    "Pollo Campero El Salvador lo que mas me encanta fue el pichel de cafe",
    "Por estar chambriando ni lo atienden a uno",
    "me encanta",
    "¡¡Eso no sirve ya!!  de café es deberían mejorar eso es horrible como le sirven agua caliente en TODOS los restaurantes",
    "Gran paja yo fui a metro al campero y le dije al que me despachó no hay hielo y me dijo si no sale es porque no hay, no es manera de hablarle a los clientes",
    "Pollo Campero El Salvador y los benedictinos???"
]

def preprocesar_texto(texto):
    texto = texto.lower()  
    texto = re.sub(r'\W', ' ', texto)  
    texto = re.sub(r'\d', ' ', texto)  
    palabras = texto.split() 
    palabras = [palabra for palabra in palabras if palabra not in stop_words]  
    return ' '.join(palabras)


mensajes_procesados = [preprocesar_texto(mensaje) for mensaje in mensajes]


vectorizador = CountVectorizer()
X = vectorizador.fit_transform(mensajes_procesados)
frecuencias = pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names_out())

frecuencia_palabras = frecuencias.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=frecuencia_palabras.head(10).index, y=frecuencia_palabras.head(10).values, palette="viridis")
plt.xlabel("Palabras")
plt.ylabel("Frecuencia")
plt.title("Frecuencia de palabras en los comentarios")
plt.xticks(rotation=45)
plt.show()
