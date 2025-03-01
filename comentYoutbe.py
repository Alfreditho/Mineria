import re
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from textblob import TextBlob
from playwright.sync_api import sync_playwright
import pandas as pd
import tkinter as tk
from tkinter import ttk

def extraer_comentarios_youtube(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        page.goto(url)
        page.wait_for_timeout(5000)

        comentarios = set()
        prev_comments_count = 0

        while True:
            nuevos_comentarios = page.locator("#content-text").all_inner_texts()
            if len(nuevos_comentarios) == prev_comments_count:
                break
            comentarios.update(nuevos_comentarios)
            prev_comments_count = len(nuevos_comentarios)
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(3000)

        browser.close()
        return list(comentarios)

# URL del video de YouTube
youtube_url = "https://www.youtube.com/watch?v=AOORf7LJWxE"
comentarios = extraer_comentarios_youtube(youtube_url)

# Lista de stopwords en español
stopwords_es = set([
    "el", "la", "los", "las", "de", "del", "y", "que", "es", "en", "un", "una", "con", "para", "muy", "me", "lo",
    "yo", "ya", "se" , "a", "mi", "o", "esta", "no", "este", "esto", "ese", "esa", "esas", "eso", "así", "tan", "cómo", "porque"
])

# Preprocesamiento de los comentarios
procesados = []
for comentario in comentarios:
    comentario = comentario.lower()
    comentario = re.sub(r"[^\w\s]", "", comentario)
    palabras = comentario.split()
    palabras_limpias = [word for word in palabras if word not in stopwords_es]
    procesados.append(" ".join(palabras_limpias))

# Análisis de sentimiento con TextBlob
sentimientos = [TextBlob(com).sentiment.polarity for com in procesados]

def clasificar_sentimiento(polaridad):
    if polaridad > 0.5:
        return "Felicidad"
    elif 0.2 < polaridad <= 0.5:
        return "Alegría"
    elif -0.2 <= polaridad <= 0.2:
        return "Neutralidad"
    elif -0.5 <= polaridad < -0.2:
        return "Tristeza"
    elif polaridad < -0.5:
        return "Ira"
    else:
        return "Miedo"

emociones = [clasificar_sentimiento(p) for p in sentimientos]
conteo_emociones = Counter(emociones)

todas_palabras = " ".join(procesados).split()
conteo_palabras = Counter(todas_palabras).most_common(10)

# Crear interfaz gráfica
root = tk.Tk()
root.title("Análisis de Sentimientos en YouTube")
root.geometry("700x700")

frame = ttk.Frame(root, padding=10)
frame.pack(fill="both", expand=True)

# Tabla de comentarios extraídos
label_comentarios = ttk.Label(frame, text="Comentarios Extraídos de YouTube", font=("Arial", 12))
label_comentarios.pack(pady=10)

tree_comentarios = ttk.Treeview(frame, columns=("Comentario"), show="headings")
tree_comentarios.heading("Comentario", text="Comentario")
tree_comentarios.pack(fill="both", expand=True)

for comentario in comentarios:
    tree_comentarios.insert("", "end", values=(comentario,))

# Tabla de palabras más repetidas
label_palabras = ttk.Label(frame, text="Palabras más repetidas", font=("Arial", 12))
label_palabras.pack(pady=10)

tree_palabras = ttk.Treeview(frame, columns=("Palabra", "Frecuencia"), show="headings")
tree_palabras.heading("Palabra", text="Palabra")
tree_palabras.heading("Frecuencia", text="Frecuencia")
tree_palabras.pack(fill="both", expand=True)

for palabra, frecuencia in conteo_palabras:
    tree_palabras.insert("", "end", values=(palabra, frecuencia))

# Graficar emociones detectadas
def mostrar_grafico():
    plt.figure(figsize=(8, 5))
    plt.bar(conteo_emociones.keys(), conteo_emociones.values(), color=["green", "yellow", "gray", "blue", "red", "purple"])
    plt.xlabel("Emoción")
    plt.ylabel("Cantidad de comentarios")
    plt.title("Distribución de emociones en comentarios de YouTube")
    plt.show()

btn_grafico = ttk.Button(frame, text="Mostrar Gráfico", command=mostrar_grafico)
btn_grafico.pack(pady=10)

root.mainloop()