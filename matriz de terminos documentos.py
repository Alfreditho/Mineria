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
    "Tiene toda la razón. Ellos ya sabían por eso pusieron esa restricción. Pero el presidente debe tomar provecho de las minas. Porque cuando él salga los países extranjeros van a hacerlo ellos y no les va a importar nada más que el dinero.",
    "De acuerdo con lo que dice el Presidente Bukele, hay que aprovechar los recursos naturales, más aún si eso va a ayudar con el desarrollo del país! Claro que sí!",
    "Solo puedo decir WOW!!!! Impresionante Presidente que llegó a quitarle la ignorancia a su gente!!",
    "Discurso fuera de lugar, se nota una ambición no a favor de la población.",
    "Un capo la verdad, me saco el sombrero.",
    "Aquí en Guatemala explotan las minas, contaminan los ríos y le dejan el 1 % de regalías a Guatemala. La riqueza se queda en los políticos y en las empresas extranjeras.",
    "La pobreza está en la mente. La pobreza no es falta de dinero, sino de ideas.",
    "Qué presidente más sabio e inteligente.",
    "Canadá no solo tiene minas en Canadá, sino que las minas de Guatemala son explotadas por Canadá, se llevan el material y pagan 1% al Estado.",
    "Bukele ha sido un buen presidente, pero aquí sí: NO a la minería, NO a la destrucción de nuestro lindo El Salvador.",
    "Pero es que se ha salido del tema para convencer. Empezó hablando bien. Dijo que los otros países extraen y grandes potencias también, pero que no sabe cómo. Entonces, ¿cómo quiere extraer acá si no sabe cómo?",
    "Recordemos que nuestro presidente es responsable, inteligente, profesional. Yo tengo la plena confianza de que va a manejar bien la minería. Nunca hemos tenido un presidente como él.",
    "Ojalá que si lo hacen protejan a la gente. Conociendo El Salvador, solo nos quedarán los enfermos y daños.",
    "El Salvador es pequeño, no vale la pena dañar el medio ambiente.",
    "Qué temor, Dios mío, que no se lleve a cabo esta destrucción.",
    "NO a la minería!!!",
    "La salud de los habitantes vale más que el oro.",
    "Qué gran forma de pensar, a buen entendedor pocas palabras. Lo que quiere es hacer crecer a su país también, no solo ser un país tercermundista y de consumismo, sino tener tecnología y hacerse poco a poco una potencia.",
    "No existe extracción minera limpia, el daño lo van a hacer y eso es irreversible.",
    "Pensé que era el presidente mejor del mundo, pero por favor NO a la minería, no la necesitamos. ¿Por qué tanta ambición?",
    "Excelente análisis, Nayib Bukele.",
    "Presidente Bukele, en el Cal 9 de La Unión no hay libertad de expresión, no se puede opinar, tienen cerrado para mandar mensaje.",
    "Porque el presidente dijo exportar pobreza para Estados Unidos y que trabajen y manden remesas???? Me siento indignado con las palabras del señor presidente. Ojalá haya una explicación por esas palabras porque el pueblo salvadoreño no se merece esos comentarios.",
    "Que Dios tenga misericordia de las personas que perderán sus tierras porque dan lo que ellos quieren dar, no pagan el precio justo. Eso le ha pasado a miles cuando construyen carreteras y les quitan sus terrenos.",
    "Por eso te reelegimos, Dios te bendiga Bukele. Confiamos en ti.",
    "Dios te salve patria sagrada. ¿Qué le espera a El Salvador? Los pocos recursos que tenemos, este señor se los quiere acabar y contaminar.",
    "A este tipo solo le importa el dinero, pero menos la salud y el bienestar de las personas. El Salvador es un país muy pequeño para tanta contaminación. Esa contaminación llegará hasta Guatemala y Honduras.",
    "El bachiller solo está interesado en el dinero, no le importa la contaminación ni la salud. El pueblo le dio el poder, él ya consiguió su objetivo.",
    "En Guatemala hay oro y otros minerales, pero siempre los corruptos hacen contratos millonarios con mineras de otros países.",
    "Bukele, el mejor.",
    "Italian, jejeje. Ay mi presi, usted no le ha dado vuelta a toda Italia para que vea la contaminación…",
    "La minería es necesaria para el desarrollo, eso sí, que sea una minería responsable con el medio ambiente. Espero que Nayib tenga las consideraciones respectivas. Si lo hace así, El Salvador será una gran nación.",
    "Ellos se van a ir después de que lo extraigan, solo van a dejar la gran contaminación y miseria.",
    "Siempre me preguntaba qué ganaban otros países con ver a El Salvador mal. Ahora todo tiene sentido.",
    "Este se quiere quebrar a todo el país.",
    "NO a la minería...",
    "Dice cómo crece la economía de China, ajá, y no sabe este señor que China es uno de los países más contaminados del mundo.",
    "Dios bendiga esa tierra.",
    "¿Por qué primeramente antes de lanzar este proyecto no se expone a la población el impacto ambiental que esto traería y los riesgos a la salud, de acuerdo a estudios previos en minas que ya existieron?",
    "La pregunta sería también... ¿Por qué en su momento se prohibió la minería en El Salvador? Que se dé una explicación transparente para que la población tenga un criterio amplio y objetivo sobre la implementación de estos proyectos en nuestro país y a quién realmente beneficiaría. Debemos primero informarnos antes de tomar decisiones y no arrepentirnos a futuro.",
    "Es que no es de limpiar ríos, ¡es dejar de contaminarlos!",
    "Gracias por ser el presidente de El Salvador. Que Dios lo bendiga y proteja siempre.",
    "Cuando no era presidente, se oponía a la minería, se apegaba a lo que defendía su padre. Hoy quiere hacer lo contrario.",
    "¿Y pondrá en reserva lo que se produzca o se gane, como el Bitcoin?",
    "Es que tiene mucha razón... Vamos por otros 5 años, presidente. Una lástima que muchas personas no tienen la visión de crecer y desarrollarse. Lo mejor es dejar de ser un país consumista. Debemos crecer.",
    "Siempre y cuando el beneficio sea para todos y con el menor daño ambiental posible, no hay problema.",
    "Pero la deforestación, contaminación de agua, migración de fauna...",
    "Pero los hondureños están más pobres que nosotros y tienen minas, y se vienen a trabajar para El Salvador. Los mexicanos también se van para Estados Unidos. Se quedan con esta riqueza los políticos y las grandes empresas, y comen menos huevos que nosotros.",
    "Solo les limpiará el agua y nada más, no esperen nada más.",
    "La minería va a ayudar a la economía de El Salvador...",
    "El problema de la minería es que para el pobre de a pie no llega ni un centavo. Ese es el gran problema, señores. Se van a acabar el país: ríos, aire, aguas subterráneas, etc., y el pobre seguirá pobre.",
    "Pacific Green mete de nuevo sus máquinas para extraer oro y uranio. Solo una pregunta: ¿qué beneficios en los pueblos ha traído la extracción de gas en El Salvador? No se ha visto, presi.",
    "Qué buen presidente Nayib Bukele.",
    "¿Para qué pregunta? Él es el rey de El Salvador, puede hacer lo que quiera.",
    "Primero Dios nos abra el entendimiento para confiar, así como hicieron con la seguridad.",
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
