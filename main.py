import pandas as pd
from openai import OpenAI
import base64
from flask import Flask, request
import os
from fuzzywuzzy import fuzz
import re
import unidecode
from io import StringIO
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (solo local)
load_dotenv()

# Obtener la clave de entorno
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está definida.")

client = OpenAI(api_key=api_key)

def cargar_base_de_datos_contenido(contenido_csv):
    try:
        df = pd.read_csv(StringIO(contenido_csv))
        if not all(col in df.columns for col in ['producto', 'precio', 'stock']):
            raise ValueError("El CSV no tiene las columnas necesarias: producto, precio, stock.")
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
        df.dropna(subset=['producto', 'precio', 'stock'], inplace=True)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Error al cargar la base de datos: {e}")
        return None

def extraer_texto_de_imagen_api(ruta_imagen):
    try:
        with open(ruta_imagen, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extrae todos los elementos de la lista escolar de la imagen. No resumas ni cortes la información, incluye todos los ítems presentes en la imagen."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    return texto

def comparar_items_con_precios(lista_extraida, base_de_datos, score_threshold=85):
    resultados = []
    base_de_datos['producto_normalizado'] = base_de_datos['producto'].apply(normalizar_texto)

    for item in lista_extraida:
        item_limpio = normalizar_texto(item)
        mejor_score = 0
        mejor_fila = None

        for idx, datos in base_de_datos.iterrows():
            prod_norm = datos['producto_normalizado']
            score = fuzz.token_set_ratio(item_limpio, prod_norm)
            if score > mejor_score:
                mejor_score = score
                mejor_fila = datos

        if mejor_fila is not None and mejor_score >= score_threshold:
            precio_val = mejor_fila['precio']
            stock_val = mejor_fila['stock']
            producto_csv = mejor_fila['producto']
            precio_final = precio_val if stock_val > 0 else "No hay stock"
            resultados.append({
                "producto_original": item,
                "producto_csv": producto_csv,
                "precio": precio_final,
                "stock": stock_val,
                "credibilidad": mejor_score
            })
        else:
            resultados.append({
                "producto_original": item,
                "producto_csv": "",
                "precio": "No hay stock",
                "stock": 0.0,
                "credibilidad": mejor_score
            })

    return resultados

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return """
    <h1>Sube una imagen de tu lista escolar y el CSV de productos</h1>
    <p>El archivo CSV debe contener las columnas: <strong>producto, precio, stock</strong></p>
    <form action='/procesar' method='post' enctype='multipart/form-data'>
        <label>Sube la imagen (PNG/JPG):</label><br>
        <input type='file' name='imagen' required><br><br>

        <label>Sube el archivo CSV (con columnas producto, precio y stock):</label><br>
        <input type='file' name='csvfile' required><br><br>

        <input type='submit' value='Subir y Procesar'>
    </form>
    """

@app.route("/procesar", methods=["POST"])
def procesar():
    if "imagen" not in request.files:
        return "Error: No se ha subido la imagen.", 400
    archivo_imagen = request.files["imagen"]
    if archivo_imagen.filename == "":
        return "Error: No se seleccionó ninguna imagen.", 400

    if "csvfile" not in request.files:
        return "Error: No se ha subido el archivo CSV.", 400
    archivo_csv = request.files["csvfile"]
    if archivo_csv.filename == "":
        return "Error: No se seleccionó ningún archivo CSV.", 400

    ruta_imagen = os.path.join(UPLOAD_FOLDER, archivo_imagen.filename)
    archivo_imagen.save(ruta_imagen)

    contenido_csv = archivo_csv.read().decode("utf-8")
    base_de_datos = cargar_base_de_datos_contenido(contenido_csv)
    if base_de_datos is None:
        return "Error al cargar la base de datos. Asegúrese de que el CSV tenga las columnas producto, precio y stock.", 500

    texto_extraido = extraer_texto_de_imagen_api(ruta_imagen)
    if texto_extraido is None:
        return "No se pudo extraer el texto de la imagen.", 500

    lista_items = [line.strip() for line in texto_extraido.split('\n') if line.strip()]
    resultados = comparar_items_con_precios(lista_items, base_de_datos, score_threshold=85)

    html_resultados = "<h2>Resultados</h2><table border='1'><tr><th>Producto Original</th><th>Producto Coincidencia</th><th>Precio</th><th>Stock</th><th>Credibilidad</th></tr>"
    for r in resultados:
        html_resultados += f"<tr><td>{r['producto_original']}</td><td>{r['producto_csv']}</td><td>{r['precio']}</td><td>{r['stock']}</td><td>{r['credibilidad']}</td></tr>"
    html_resultados += "</table>"

    return html_resultados

if __name__ == "__main__":
    app.run(debug=True)
