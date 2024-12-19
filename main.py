import time
import pandas as pd
from openai import OpenAI
import base64
from flask import Flask, request, g
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

# Decorador para medir el tiempo de ejecución de las funciones
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        if not hasattr(g, 'timings'):
            g.timings = []
        g.timings.append((func.__name__, duration))
        return result
    return wrapper

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variable de entorno OPENAI_API_KEY no está definida.")

client = OpenAI(api_key=api_key)

@measure_time
def cargar_base_de_datos_contenido(contenido_csv):
    try:
        df = pd.read_csv(StringIO(contenido_csv))
        # Verificar que todas las columnas necesarias estén presentes, incluyendo 'SKU'
        columnas_necesarias = ['producto', 'precio', 'stock', 'SKU']
        if not all(col in df.columns for col in columnas_necesarias):
            raise ValueError(f"El CSV no tiene las columnas necesarias: {', '.join(columnas_necesarias)}.")
        df['precio'] = pd.to_numeric(df['precio'], errors='coerce')
        df['stock'] = pd.to_numeric(df['stock'], errors='coerce')
        df.dropna(subset=['producto', 'precio', 'stock', 'SKU'], inplace=True)
        return df.reset_index(drop=True)
    except Exception as e:
        print(f"Error al cargar la base de datos: {e}")
        return None

@measure_time
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
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None

@measure_time
def extraer_texto_de_pdf(ruta_pdf):
    texto = ""
    try:
        with pdfplumber.open(ruta_pdf) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texto += page_text + "\n"
        return texto.strip()
    except Exception as e:
        print(f"Error al procesar el PDF: {e}")
        return None

@measure_time
def extraer_texto_de_docx(ruta_docx):
    try:
        doc = Document(ruta_docx)
        texto = ""
        for p in doc.paragraphs:
            texto += p.text + "\n"
        return texto.strip()
    except Exception as e:
        print(f"Error al procesar el DOCX: {e}")
        return None

def normalizar_texto(texto):
    texto = texto.lower()
    texto = unidecode.unidecode(texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = texto.strip()
    return texto

def procesar_respuesta_json(contenido):
    # El contenido devuelto por la API debería ser un JSON con la estructura pedida
    # Puede contener delimitadores ``` o ```json, los removemos.
    contenido_limpio = contenido.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(contenido_limpio)
        return data
    except json.JSONDecodeError as e:
        print(f"Error al decodificar JSON: {e}")
        print(f"Contenido problemático:\n{contenido_limpio}")
        return None

@measure_time
def extraer_items_de_texto_con_openai(texto):
    """
    Esta función ahora no solo extrae los ítems, sino que también los interpreta (cantidad).
    Se realiza en un solo prompt para evitar el segundo llamado.
    """
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
    except Exception as e:
        print(f"Error al procesar el texto con OpenAI: {e}")
        return None

@measure_time
def comparar_items_con_precios(lista_items, base_de_datos, score_threshold=55):
    """
    lista_items ahora es una lista de dict con {producto_original, cantidad}.
    """
    resultados = []
    base_de_datos['producto_normalizado'] = base_de_datos['producto'].apply(normalizar_texto)

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
            if precio_unitario != "No hay stock":
                precio_total = precio_unitario * cantidad
            else:
                precio_total = "-"
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
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
@measure_time
def index():
    return """
    <h1>Sube un archivo (PNG/JPG/PDF/DOCX) y el CSV de productos</h1>
    <p>El CSV debe tener columnas: producto, precio, stock, SKU</p>
    <form action='/procesar' method='post' enctype='multipart/form-data'>
        <label>Archivo (PNG/JPG/PDF/DOCX):</label><br>
        <input type='file' name='archivo' required><br><br>

        <label>Archivo CSV (producto, precio, stock, SKU):</label><br>
        <input type='file' name='csvfile' required><br><br>

        <input type='submit' value='Subir y Procesar'>
    </form>
    """

@app.route("/procesar", methods=["POST"])
@measure_time
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

    # Calcular total general
    total_general = 0
    for r in resultados:
        if r['precio_total'] != "-" and isinstance(r['precio_total'], (int, float)):
            total_general += r['precio_total']

    html_resultados = """
    <h2>Resultados</h2>
    <table border='1' style='border-collapse:collapse;width:100%;font-family:Arial;font-size:14px;'>
    <tr style='background:#f0f0f0;'>
      <th style='padding:5px;'>Producto Original</th>
      <th style='padding:5px;'>Producto Coincidencia</th>
      <th style='padding:5px;'>SKU</th>
      <th style='padding:5px;'>Cantidad</th>
      <th style='padding:5px;'>Precio Unitario</th>
      <th style='padding:5px;'>Precio Total</th>
      <th style='padding:5px;'>Stock</th>
      <th style='padding:5px;'>Credibilidad</th>
    </tr>
    """

    for r in resultados:
        row_style = "background:#ffe6e6;" if r['producto_csv'] == "Sin coincidencias" else ""
        html_resultados += f"""
        <tr style='{row_style}'>
          <td style='padding:5px;'>{r['producto_original']}</td>
          <td style='padding:5px;'>{r['producto_csv']}</td>
          <td style='padding:5px;'>{r['SKU']}</td>
          <td style='padding:5px;'>{r['cantidad']}</td>
          <td style='padding:5px;'>{r['precio_unitario']}</td>
          <td style='padding:5px;'>{r['precio_total']}</td>
          <td style='padding:5px;'>{r['stock']}</td>
          <td style='padding:5px;'>{r['credibilidad']}</td>
        </tr>
        """

    html_resultados += f"""
    <tr style='background:#d0ffd0;'>
      <td colspan='5' style='padding:5px;text-align:right;font-weight:bold;'>Total General:</td>
      <td style='padding:5px;font-weight:bold;'>{total_general}</td>
      <td colspan='2'></td>
    </tr>
    </table>
    """

    # Imprimir los tiempos de ejecución en la consola
    if hasattr(g, 'timings'):
        print("Tiempos de ejecución por función:")
        for func_name, duration in g.timings:
            print(f"{func_name}: {duration:.4f} segundos")

    return html_resultados

if __name__ == "__main__":
    app.run(debug=True)
