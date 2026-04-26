"""
app.py
------
Servidor Flask — Motor de inferencia + API REST para la interfaz web.

Endpoints:
  GET  /           → Sirve la página principal (index.html)
  POST /classify   → Recibe JSON {subject, description}, retorna predicción + probabilidades
  GET  /health     → Estado del servidor y modelo
"""

import os
import sys
import json

from flask import Flask, render_template, request, jsonify

# Agregar src/ al path para importar los módulos del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from preprocessor import preprocess
from naive_bayes  import NaiveBayesClassifier


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'modelo_nb.pkl')

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Íconos y colores por categoría (usados en el frontend)
# ---------------------------------------------------------------------------
CATEGORY_META = {
    'ACCOUNT':      {'icon': 'fa-user-circle',            'color': '#2563eb', 'bg': '#eff6ff'},
    'CANCEL':       {'icon': 'fa-ban',                    'color': '#ea580c', 'bg': '#fff7ed'},
    'CONTACT':      {'icon': 'fa-headset',                'color': '#0891b2', 'bg': '#ecfeff'},
    'DELIVERY':     {'icon': 'fa-truck',                  'color': '#16a34a', 'bg': '#f0fdf4'},
    'FEEDBACK':     {'icon': 'fa-star',                   'color': '#ca8a04', 'bg': '#fefce8'},
    'INVOICE':      {'icon': 'fa-file-invoice',           'color': '#7c3aed', 'bg': '#f5f3ff'},
    'ORDER':        {'icon': 'fa-box',                    'color': '#0284c7', 'bg': '#f0f9ff'},
    'PAYMENT':      {'icon': 'fa-credit-card',            'color': '#059669', 'bg': '#ecfdf5'},
    'REFUND':       {'icon': 'fa-rotate-left',            'color': '#dc2626', 'bg': '#fef2f2'},
    'SHIPPING':     {'icon': 'fa-shipping-fast',          'color': '#d97706', 'bg': '#fffbeb'},
    'SUBSCRIPTION': {'icon': 'fa-calendar-check',         'color': '#9333ea', 'bg': '#faf5ff'},
}

# ---------------------------------------------------------------------------
# Carga del modelo al iniciar el servidor
# ---------------------------------------------------------------------------
classifier = NaiveBayesClassifier()

def load_model():
    if os.path.exists(MODEL_PATH):
        classifier.load(MODEL_PATH)
        print(f"[app] Modelo cargado. Clases: {classifier.classes}")
        return True
    else:
        print(f"[app] ADVERTENCIA: modelo no encontrado en {MODEL_PATH}")
        print("[app] Ejecuta primero:  python src/train.py")
        return False

model_ready = load_model()


# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if not model_ready:
        return jsonify({'error': 'Modelo no disponible. Ejecuta python src/train.py primero.'}), 503

    data        = request.get_json(force=True)
    subject     = data.get('subject', '').strip()
    description = data.get('description', '').strip()

    if not subject and not description:
        return jsonify({'error': 'Debes ingresar al menos el asunto o descripción del ticket.'}), 400

    # Preprocesar y clasificar
    text        = f"{subject} {description}"
    tokens      = preprocess(text)

    if not tokens:
        return jsonify({'error': 'No se pudo extraer información útil del texto ingresado.'}), 400

    predicted, probabilities = classifier.predict_proba(tokens)

    # Enriquecer respuesta con metadatos visuales
    result_proba = {}
    for cls, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        result_proba[cls] = {
            'probability': round(prob * 100, 2),
            'icon':        CATEGORY_META.get(cls, {}).get('icon',  'fa-tag'),
            'color':       CATEGORY_META.get(cls, {}).get('color', '#374151'),
            'bg':          CATEGORY_META.get(cls, {}).get('bg',    '#f9fafb'),
        }

    return jsonify({
        'prediction':    predicted,
        'probabilities': result_proba,
        'meta':          CATEGORY_META.get(predicted, {}),
        'token_count':   len(tokens),
    })


@app.route('/health')
def health():
    return jsonify({
        'status':      'ok' if model_ready else 'model_missing',
        'model_path':  MODEL_PATH,
        'classes':     classifier.classes if model_ready else [],
        'vocab_size':  len(classifier.vocabulary) if model_ready else 0,
    })


# ---------------------------------------------------------------------------
# INICIO
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Mesa de Ayuda — Clasificador de Tickets")
    print("  Universidad Rafael Landívar · IA 2026")
    print("="*55)
    app.run(debug=True, port=5000)
