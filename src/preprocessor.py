"""
preprocessor.py
---------------
Preprocesamiento de texto para el clasificador Naive Bayes.
Pasos: lowercase -> limpieza -> tokenizacion -> eliminacion de stopwords -> stemming
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Descarga de recursos NLTK necesarios (solo la primera vez)
nltk.download('punkt',      quiet=True)
nltk.download('punkt_tab',  quiet=True)
nltk.download('stopwords',  quiet=True)

# Stopwords en ingles: "the", "is", "at", "which", etc.
STOP_WORDS = set(stopwords.words('english'))

# PorterStemmer reduce palabras a su raiz morfologica
# Ejemplo: "billing" -> "bill", "cancelling" -> "cancel", "shipped" -> "ship"
STEMMER = PorterStemmer()


def preprocess(text: str) -> list:
    """
    Limpia y tokeniza un texto aplicando todos los pasos del pipeline.

    Args:
        text: cadena de texto cruda (solicitud del cliente)

    Returns:
        Lista de tokens preprocesados (stems) listos para el modelo
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1. Convertir a minusculas para normalizar
    text = text.lower()

    # 2. Eliminar placeholders del tipo {{Order Number}}, {{Account Number}}, etc.
    # El dataset Bitext incluye estos marcadores en el texto que no aportan informacion
    text = re.sub(r'\{\{.*?\}\}', ' ', text)

    # 3. Eliminar URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 4. Eliminar caracteres especiales, digitos y puntuacion
    # Solo se conservan letras del alfabeto ingles y espacios
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 5. Colapsar multiples espacios en uno solo
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Tokenizar el texto en palabras individuales
    tokens = word_tokenize(text)

    # 7. Eliminar stopwords y tokens muy cortos, luego aplicar stemming
    # Solo se conservan tokens de mas de 2 caracteres que no sean stopwords
    tokens = [
        STEMMER.stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return tokens
