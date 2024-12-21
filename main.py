import time
import pandas as pd
from openai import OpenAI
import base64
from flask import Flask, request, g, render_template, redirect, url_for, session, flash
import os
from fuzzywuzzy import fuzz
import re
import unidecode
from io import StringIO
from dotenv import load_dotenv
import pdfplumber
from docx import Document
import json
from functools import wraps
from datetime import timedelta

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash("Por favor, inicia sesión para acceder a esta página.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está definida.")

client = OpenAI(api_key=api_key)

def cargar_base_de_datos_contenido(contenido_csv):
    try:
        df = pd.read_csv(StringIO(contenido_csv))
        columnas_necesarias = ['producto', 'precio', 'stock', 'SKU']
        if not all(col in df.columns for col in columnas_necesarias):
            raise ValueError(f"El CSV no tiene las columnas necesarias: {', '.join(columnas_necesarias)}.")
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
        df.dropna(subset=['producto', 'precio', 'stock', 'SKU'], inplace=True)
        return df.reset_index(drop=True)
    except:
        return None

def extraer_texto_de_imagen_api(ruta_imagen):
    try:
        with open(ruta_imagen, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extrae todos los elementos de la lista escolar de la imagen. Cada ítem en una línea separada con un '-'. Después, determina la cantidad interpretando el texto original. Si no se encuentra una cantidad explícita, asume 1. Devuelve un JSON con la estructura:\n[\n  {\"producto_original\": \"...\", \"cantidad\": <numero>},\n  ...\n]."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0
            )
            contenido = response.choices[0].message.content.strip()
            return procesar_respuesta_json(contenido)
    except:
        return None

def extraer_texto_de_pdf(ruta_pdf):
    try:
        texto = ""
        with pdfplumber.open(ruta_pdf) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texto += page_text + "\n"
        return texto.strip()
    except:
        return None

def extraer_texto_de_docx(ruta_docx):
    try:
        doc = Document(ruta_docx)
        texto = ""
        for p in doc.paragraphs:
            texto += p.text + "\n"
        return texto.strip()
    except:
        return None

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    return texto

def procesar_respuesta_json(contenido):
    contenido_limpio = contenido.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(contenido_limpio)
        return data
    except:
        return None

def extraer_items_de_texto_con_openai(texto):
    try:
        prompt = (
            "Tienes que extraer la lista de productos/items escolares exclusivamente del siguiente texto. "
            "No resumas ni cortes la información, incluye todos los ítems presentes. "
            "Lista cada ítem en una línea separada precedido por '-', y luego interpreta las cantidades. "
            "Si no hay cantidad explícita, asume 1.\n"
            "Devuelve un JSON con la estructura:\n"
            "[\n"
            "  {\"producto_original\": \"...\", \"cantidad\": <numero>},\n"
            "  ...\n"
            "]\n\n"
            f"Texto:\n{texto}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0
        )
        contenido = response.choices[0].message.content.strip()
        return procesar_respuesta_json(contenido)
    except:
        return None

def comparar_items_con_precios(lista_items, base_de_datos, score_threshold=55):
    base_de_datos['producto_normalizado'] = base_de_datos['producto'].apply(normalizar_texto)
    resultados = []

    for item_data in lista_items:
        producto_original = item_data['producto_original']
        cantidad = item_data['cantidad']
        item_limpio = normalizar_texto(producto_original)
        mejor_score = 0
        mejor_fila = None

        for idx, datos in base_de_datos.iterrows():
            prod_norm = datos['producto_normalizado']
            score = fuzz.token_set_ratio(item_limpio, prod_norm)
            if score > mejor_score:
                mejor_score = score
                mejor_fila = datos

        if mejor_fila is not None and mejor_score >= score_threshold:
            precio_unitario = mejor_fila['precio'] if mejor_fila['stock'] > 0 else "No hay stock"
            precio_total = precio_unitario * cantidad if precio_unitario != "No hay stock" else "-"
            resultados.append({
                "producto_original": producto_original,
                "producto_csv": mejor_fila['producto'],
                "SKU": mejor_fila['SKU'],
                "cantidad": cantidad,
                "precio_unitario": precio_unitario,
                "precio_total": precio_total,
                "stock": mejor_fila['stock'],
                "credibilidad": mejor_score
            })
        else:
            resultados.append({
                "producto_original": producto_original,
                "producto_csv": "Sin coincidencias",
                "SKU": "-",
                "cantidad": cantidad,
                "precio_unitario": "-",
                "precio_total": "-",
                "stock": "-",
                "credibilidad": mejor_score
            })

    return resultados

app = Flask(__name__)
app.secret_key = 'LbX6pxQ8bW3uEdWbLAoPjUregZbgPNg3'
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Establecer duración de la sesión a 4 horas
app.permanent_session_lifetime = timedelta(hours=4)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "dhgroup2024" and password == "dhgroup2024":
            session['logged_in'] = True
            session.permanent = True
            flash("Has iniciado sesión exitosamente.", "success")
            return redirect(url_for('index'))
        else:
            flash("Usuario o contraseña incorrectos.", "danger")
            return redirect(url_for('login'))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    session.pop('logged_in', None)
    flash("Has cerrado sesión.", "success")
    return redirect(url_for('login'))

@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template("index.html")

@app.route("/procesar", methods=["POST"])
@login_required
def procesar():
    if "archivo" not in request.files:
        return "Error: No se ha subido el archivo.", 400
    archivo = request.files["archivo"]
    if archivo.filename == "":
        return "Error: No se seleccionó ningún archivo.", 400

    if "csvfile" not in request.files:
        return "Error: No se subió el CSV.", 400
    csvfile = request.files["csvfile"]
    if csvfile.filename == "":
        return "Error: No se seleccionó ningún CSV.", 400

    ruta_archivo = os.path.join(UPLOAD_FOLDER, archivo.filename)
    archivo.save(ruta_archivo)

    contenido_csv = csvfile.read().decode("utf-8")
    base_de_datos = cargar_base_de_datos_contenido(contenido_csv)
    if base_de_datos is None:
        return "Error al cargar la base de datos.", 500

    extension = os.path.splitext(archivo.filename)[1].lower()

    if extension in [".png", ".jpg", ".jpeg"]:
        items_interpretados = extraer_texto_de_imagen_api(ruta_archivo)
    elif extension == ".pdf":
        texto_extraido = extraer_texto_de_pdf(ruta_archivo)
        if not texto_extraido:
            return "No se pudo extraer texto del PDF.", 500
        items_interpretados = extraer_items_de_texto_con_openai(texto_extraido)
    elif extension == ".docx":
        texto_extraido = extraer_texto_de_docx(ruta_archivo)
        if not texto_extraido:
            return "No se pudo extraer texto del DOCX.", 500
        items_interpretados = extraer_items_de_texto_con_openai(texto_extraido)
    else:
        return "Tipo de archivo no soportado. Debe ser PNG, JPG, PDF o DOCX.", 400

    if items_interpretados is None or not isinstance(items_interpretados, list):
        return "No se pudo extraer e interpretar ítems con OpenAI.", 500

    resultados = comparar_items_con_precios(items_interpretados, base_de_datos, score_threshold=55)
    total_general = 0
    for r in resultados:
        if r['precio_total'] != "-" and isinstance(r['precio_total'], (int, float)):
            total_general += r['precio_total']

    datos_para_template = {
        "resultados": resultados,
        "total_general": total_general
    }

    return render_template("resultados.html", **datos_para_template)

if __name__ == "__main__":
    app.run(debug=True)
