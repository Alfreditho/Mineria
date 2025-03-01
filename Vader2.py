import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Descargar recursos de nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

# Lista de comentarios integrados manualmente
mensajes = [
    "yo pienso que no deberían mostrar las caras de las personas que entrevistan, las ponen en peligro. adelante el salvador, desde olancho, honduras, algún día iré a visitarlos.",
    "los salvadoreños amamos la libertad.",
    "los que no opinan es porque tienen la cola prendada.",
    "los que dijeron que no pueden opinar sobre eso es porque están dolidos, porque sus niños ya se los llevaron y no les van a traer dinero de las extorsiones. están en contra de la medida, pero también tienen miedo de decir cosas porque saben que se van a quemar solitos.",
    "excelente, sencillamente excelente. 👍👌 es que creo que no hay otra palabra mejor que está. hoy sí se siente la seguridad en nuestro país. todos nos merecemos un buen país. fuera antisociales, fuera fmln, fuera arena, fuera todo aquel que no apoya a este bello y grandioso gobierno. el que no está de acuerdo es parte de la familia de estos delincuentes, y son los gorgojos de la oposición.",
    "eso es el verdadero salvadoreño honrado. se sienten como en otro planeta. bien por el salvador, adelante, ni un paso atrás, ni siquiera para agarrar impulso. dios los bendiga de parte de un nica que los aprecia enormemente.",
    "todo el año deberían de extender el régimen de excepción.",
    "todo salvadoreño honrado está de acuerdo y opina... #yoestoyconbukele 🌹👏👏👏👏👏",
    "queremos régimen de excepción todo el año, ya escucharon la voz del pueblo. estamos felices con esta medida y queremos más.",
    "adelante, sigan con el régimen de excepción hasta que esta plaga sea exterminada sin límite de tiempo.",
    "excelente gobierno, dios siga dando mucha sabiduría y bendiciones.",
    "sí que sigan así, mi gobierno y su gabinete de seguridad, fuerza armada y policías 👮‍♀️, y que sean más duros con estos delincuentes.",
    "gracias, señor presidente. dios lo bendiga. estamos con usted. gracias.",
    "no den los rostros de las personas. todavía no es seguro. por eso muchos no quieren opinar...",
    "para los civiles honrados y trabajadores, nos encanta cómo va caminando el país de la mano de nayib bukele.",
    "ahora andamos más libres con el régimen como nunca antes.",
    "todos salimos, disfrutamos de la seguridad...",
    "gracias fuerza armada por el trabajo.",
    "felicito nuestra propuesta del presidente nayib bukele, que sigan permanentemente el régimen de excepción.",
    "esas son las opiniones que ningún medio internacional da. gracias canal 12 por reportar esto! gracias, muchas gracias! estas son las opiniones que importan.",
    "felicitaciones a bukele y a la gente que lo llevó a ganar la presidencia. la verdad, mis respetos, es el mejor presidente de todo el continente americano.",
    "aplaudo el excelente trabajo que están haciendo el presidente nayib bukele, su equipo de gobierno y las instituciones que representan la ley y el orden de nuestro país. el régimen de excepción debe mantenerse vigente todo el tiempo que sea necesario para poder combatir eficazmente la delincuencia y así garantizar el bienestar y tranquilidad de nuestro pueblo.",
    "se huele el bien ahí, más limpio, libres por fin.",
    "bukele 2024.",
    "gracias a dios porque nuestro presidente fue enviado por dios. eso es lo que necesitábamos y lo necesitamos siempre, que toda persona que ande haciendo lo malo será castigada por la ley.",
    "es imperativo la pronta instauración de la pena capital en el salvador. solo así haremos prevalecer la justicia y un verdadero estado de derecho en el cual sea prioridad el respeto a la vida, la dignidad y el bienestar del pueblo salvadoreño.",
    "la política es bien difícil... está buena esa conclusión de la señora.",
    "y los policías que han colaborado con las maras... ¿cuándo?",
    "por lo que está haciendo el presidente, el salvador va a subir con la economía mucho más allá de lo que lo podemos imaginar. dios bendiga a nuestro presidente y su familia.",
    "cuando uno va de vacaciones allá, pero ni dormir se puede, pensando que se van a meter a las casas a robar, porque a mí ya me robaron mucho dinero allá.",
    "de todas las periodistas y locutoras de el salvador, la mejor voz es la de jenifer flores, me gusta 😍.",
    "todo el que se oponga a esto debe de estar preso. qué bien se siente poder salir y no tener miedo.",
    "ay, periodista, no le ponga la cámara a la gente, no sea torpe.",
    "qué mal que no protejan el rostro a las personas que entrevistan.",
    "yo estoy de acuerdo, pero que se lleven a los pandilleros, pero ahora se llevan a la gente que no deben. ¡eso no está bien!",
    "¿y los jóvenes que opinan? ¿o es que la opinión de la juventud no cuenta? okey.",
    "ya solo faltan los de clase media y alta... hay tamales allí también.",
    "es terrible que sigan con el régimen de excepción, ya basta de tanta represión.",
    "me da miedo pensar que estamos perdiendo nuestra libertad bajo el pretexto de la seguridad.",
    "no creo que esto esté funcionando. más represión no es la solución, solo nos están silenciando.",
    "no estoy de acuerdo con que se lleven a todas las personas. esto está afectando a gente inocente.",
    "me preocupa que se siga criminalizando a todo el pueblo, no solo a los pandilleros.",
    "qué bonito es ver cómo el país mejora, ojalá sigan así para que nuestros hijos puedan vivir mejor.",
    "estoy emocionado de ver cómo avanza el país, me siento seguro y orgulloso de ser salvadoreño.",
    "es una bendición vivir en un país con tanta seguridad, me siento como nunca antes, libre.",
    "hace unos años no pensábamos que veríamos tanta paz. gracias, presidente bukele, por hacernos sentir tranquilos.",
    "gracias a dios por este gobierno, finalmente veo un futuro mejor para mi familia y yo.",
    "me asusta pensar en las consecuencias de este régimen. ¿qué pasará si no logramos terminar con las pandillas?",
    "aunque veo mejoras, sigo temiendo que los derechos humanos se vean comprometidos por estas medidas.",
    "estoy feliz por la seguridad, pero temo que esto sea solo un parche y que los problemas continúen.",
    "me preocupa que la gente esté comenzando a perder su libertad para sentirse segura, no es justo.",
    "tengo miedo de ser detenido injustamente solo por vivir en un barrio vulnerable.",
    "no puedo dejar de sentir tristeza por aquellos que han sido injustamente capturados durante este régimen.",
    "la gente ya no tiene miedo de caminar por las calles, pero temo que el precio de la seguridad sea muy alto.",
    "me siento triste al ver cómo algunas familias ya no pueden salir de sus hogares sin miedo a ser detenidos.",
    "el miedo a ser detenido por error me atormenta, ni siquiera salir a comprar pan me siento tranquilo.",
    "estoy triste porque me parece que el régimen está afectando a las personas que realmente no tienen nada que ver con el crimen.",
    "es triste ver cómo los derechos humanos de algunos ciudadanos están siendo pisoteados bajo el pretexto de la seguridad.",
    "me siento impotente ante la situación. la libertad es un derecho que estamos perdiendo poco a poco.",
    "me da tristeza que muchos de mis vecinos ahora teman salir, ya no sé si por el régimen o por las pandillas.",
    "me siento feliz de ver cómo avanzamos, pero a veces también me da miedo que esto no dure mucho tiempo.",
    "es una alegría ver el progreso, pero ¿a qué costo? eso es lo que me pregunto cada día.",
    "los avances son impresionantes, pero el miedo a perderlos me persigue todos los días.",
    "nunca pensé que mi país estaría tan tranquilo, pero me da miedo que algo pueda cambiar en cualquier momento."
]


# Preprocesamiento de texto
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\W', ' ', texto)  # Eliminar caracteres especiales
    texto = re.sub(r'\d', ' ', texto)  # Eliminar números
    palabras = texto.split()
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    return ' '.join(palabras)

mensajes_procesados = [preprocesar_texto(mensaje) for mensaje in mensajes]

# Funciones de análisis de sentimientos
def analizar_sentimientos_texto(texto):
    analyzer = SentimentIntensityAnalyzer()
    sentimiento = analyzer.polarity_scores(texto)
    return sentimiento

def obtener_polaridad_emociones(texto):
    blob = TextBlob(texto)
    polaridad = blob.sentiment.polarity
    if polaridad > 0:
        emocion = "Positiva"
    elif polaridad < 0:
        emocion = "Negativa"
    else:
        emocion = "Neutral"
    return emocion

def clasificar_emocion(polaridad):
    if polaridad > 0.2:
        return "Alegría"
    elif polaridad > 0.2:
        return "Sorpresa"
    elif polaridad > -0.2:
        return "Neutral"
    elif polaridad > -0.2:
        return "Miedo"
    else:
        return "Tristeza"

# Aplicar análisis
resultados_vader = [analizar_sentimientos_texto(mensaje) for mensaje in mensajes_procesados]
emociones_clasificadas = [clasificar_emocion(TextBlob(mensaje).sentiment.polarity) for mensaje in mensajes_procesados]

# Visualización de datos
emociones_count = pd.Series(emociones_clasificadas).value_counts()

# Gráfico de barras para frecuencias de emociones
plt.figure(figsize=(8, 5))
sns.barplot(x=emociones_count.index, y=emociones_count.values, palette="coolwarm")
plt.xlabel("Emoción")
plt.ylabel("Cantidad")
plt.title("Distribución de Emociones en los Comentarios")
plt.show()

# Gráfico circular para proporciones de cada emoción
plt.figure(figsize=(8, 8))
plt.pie(emociones_count, labels=emociones_count.index, autopct='%1.1f%%', colors=sns.color_palette("coolwarm", len(emociones_count)))
plt.title("Proporción de Emociones en los Comentarios")
plt.show()

# Generar nube de palabras
todo_texto = ' '.join(mensajes_procesados)
nube_palabras = WordCloud(width=800, height=400, background_color='white').generate(todo_texto)
plt.figure(figsize=(10, 5))
plt.imshow(nube_palabras, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de Palabras de los Comentarios")
plt.show()
