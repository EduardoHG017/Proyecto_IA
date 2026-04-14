"""
preprocessor.py
---------------
Preprocesamiento de texto para el clasificador Naive Bayes.
Pasos: lowercase → limpieza → tokenización → eliminación de stopwords → stemming
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

STOP_WORDS = set(stopwords.words('english'))
STEMMER    = PorterStemmer()


def preprocess(text: str) -> list:
    """
    Limpia y tokeniza un texto.

    Args:
        text: cadena de texto cruda

    Returns:
        Lista de tokens preprocesados (stems)
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # 1. Convertir a minúsculas
    text = text.lower()

    # 2. Eliminar URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)

    # 3. Eliminar caracteres especiales, dígitos y puntuación — dejar solo letras y espacios
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 4. Colapsar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()

    # 5. Tokenizar
    tokens = word_tokenize(text)

    # 6. Eliminar stopwords y tokens muy cortos, luego aplicar stemming
    tokens = [
        STEMMER.stem(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]

    return tokens
