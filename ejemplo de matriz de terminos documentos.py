import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
# from textblod import TextBlod


nltk.download('stopwords')
nltk.download('wordnet')

mensajes = [
    "En metro ayer nos atendieron bien y los desayunos son una delicia",
    "Los mejores desayunos... Y sin duda el mejor cafe, mi rÃ©cord personal son 5 tazas en un desayuno!!!",
    "El servicio al cliente es excelente",
    "En el puerto de la libertad dan un pesimo servicio",
    "Pollo Campero El Salvador lo que mas me encanta fue el pichel de cafe",
    "Por estar chambriando ni lo atienden a uno",
    "me encnta",
    "¡¡Eso no sirve ya!!  de café es deberían mejorar eso es horrible como le sirven agua caliente en TODOS los restaurantes",
    "Gran paja yo fui a metro al campero y le dije al que me despachÃ³ no hay hielo y me dijo si no sale es porque no hay, no es manera de hablarle a los clientes",
    "Pollo Campero El Salvador y los benedictinos???"
]


def preprocesar_texto(texto):

    texto = texto.lower()

    texto = re.sub(r'\W', ' ', texto)
    texto = re.sub(r'\d', ' ', texto)

    palabras = texto.split()

    palabras = [palabra for palabra in palabras if palabra not in stopwords.words('spanish')]

    lematizador = WordNetLemmatizer()
    palabras = [lematizador.lemmatize(palabra) for palabra in palabras]
    return ' '.join(palabras)


mensajes_procesados = [preprocesar_texto(mensaje) for mensaje in mensajes]


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(mensajes_procesados)
mtd = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("Matriz de Términos-Documentos:")
print(mtd)
