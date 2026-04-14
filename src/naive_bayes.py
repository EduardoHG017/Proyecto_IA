"""
naive_bayes.py
--------------
Implementación manual del clasificador Naive Bayes Multinomial.

Técnicas implementadas:
  - Bag of Words (vocabulario construido desde el corpus de entrenamiento)
  - Probabilidades a priori por clase: log P(clase)
  - Verosimilitud con Laplace Smoothing: P(w|clase) = (count + 1) / (total + |V|)
  - Suma de logaritmos durante la inferencia para evitar underflow numérico
  - Guardado y carga del modelo via pickle
"""

import math
import pickle
from collections import defaultdict


class NaiveBayesClassifier:

    def __init__(self):
        self.classes            = []        # lista de clases únicas
        self.vocabulary         = set()     # vocabulario global (Bag of Words)
        self.class_priors       = {}        # log P(clase)
        self.word_log_likelihoods = {}      # {clase: {palabra: log P(palabra|clase)}}
        self.class_doc_counts   = {}        # nº de documentos por clase
        self.class_word_counts  = {}        # nº total de tokens por clase
        self.class_vocab        = {}        # {clase: {palabra: frecuencia}}
        self.total_docs         = 0

    # ------------------------------------------------------------------
    # ENTRENAMIENTO
    # ------------------------------------------------------------------
    def train(self, documents: list, labels: list) -> None:
        """
        Entrena el modelo con una lista de documentos tokenizados y sus etiquetas.

        Parámetros
        ----------
        documents : list[list[str]]   — corpus tokenizado
        labels    : list[str]         — etiqueta de clase por documento
        """
        self.classes    = list(set(labels))
        self.total_docs = len(documents)

        # Inicializar contadores
        for cls in self.classes:
            self.class_doc_counts[cls]  = 0
            self.class_word_counts[cls] = 0
            self.class_vocab[cls]       = defaultdict(int)

        # Contar ocurrencias de palabras por clase
        for tokens, label in zip(documents, labels):
            self.class_doc_counts[label] += 1
            for token in tokens:
                self.class_vocab[label][token] += 1
                self.class_word_counts[label]  += 1
                self.vocabulary.add(token)

        vocab_size = len(self.vocabulary)

        # ---- Probabilidades a priori (log) ----
        # log P(clase) = log( N_clase / N_total )
        for cls in self.classes:
            self.class_priors[cls] = math.log(
                self.class_doc_counts[cls] / self.total_docs
            )

        # ---- Verosimilitud con Laplace Smoothing (log) ----
        # log P(w|clase) = log( (count(w,clase) + 1) / (N_clase + |V|) )
        self.word_log_likelihoods = {}
        for cls in self.classes:
            self.word_log_likelihoods[cls] = {}
            denominator = self.class_word_counts[cls] + vocab_size

            for word in self.vocabulary:
                count = self.class_vocab[cls].get(word, 0)
                self.word_log_likelihoods[cls][word] = math.log(
                    (count + 1) / denominator
                )

            # Probabilidad para palabras no vistas (fuera del vocabulario)
            self.word_log_likelihoods[cls]['__OOV__'] = math.log(
                1 / denominator
            )

    # ------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------
    def predict(self, tokens: list) -> tuple:
        """
        Clasifica un documento tokenizado.

        Usa suma de logaritmos:
            log P(clase|doc) ∝ log P(clase) + Σ log P(w|clase)

        Retorna
        -------
        (clase_predicha, dict{clase: log_score})
        """
        log_scores = {}
        for cls in self.classes:
            # Iniciar con el prior
            score = self.class_priors[cls]

            # Acumular log-verosimilitudes (evita underflow)
            for token in tokens:
                if token in self.word_log_likelihoods[cls]:
                    score += self.word_log_likelihoods[cls][token]
                else:
                    score += self.word_log_likelihoods[cls]['__OOV__']

            log_scores[cls] = score

        predicted = max(log_scores, key=log_scores.get)
        return predicted, log_scores

    def predict_proba(self, tokens: list) -> tuple:
        """
        Retorna la clase predicha y las probabilidades normalizadas por clase.
        Convierte los log-scores a probabilidades con softmax numérico.
        """
        predicted, log_scores = self.predict(tokens)

        # Softmax estable: restar el máximo antes de exponenciar
        max_score  = max(log_scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in log_scores.items()}
        total      = sum(exp_scores.values())
        proba      = {cls: exp_scores[cls] / total for cls in exp_scores}

        return predicted, proba

    # ------------------------------------------------------------------
    # PERSISTENCIA
    # ------------------------------------------------------------------
    def save(self, filepath: str) -> None:
        """Serializa el modelo entrenado en un archivo pickle."""
        model_data = {
            'classes':               self.classes,
            'vocabulary':            self.vocabulary,
            'class_priors':          self.class_priors,
            'word_log_likelihoods':  self.word_log_likelihoods,
            'class_doc_counts':      self.class_doc_counts,
            'class_word_counts':     self.class_word_counts,
            'total_docs':            self.total_docs,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"[NaiveBayes] Modelo guardado en: {filepath}")

    def load(self, filepath: str) -> None:
        """Carga un modelo previamente guardado desde un archivo pickle."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.classes              = model_data['classes']
        self.vocabulary           = model_data['vocabulary']
        self.class_priors         = model_data['class_priors']
        self.word_log_likelihoods = model_data['word_log_likelihoods']
        self.class_doc_counts     = model_data['class_doc_counts']
        self.class_word_counts    = model_data['class_word_counts']
        self.total_docs           = model_data['total_docs']
        print(f"[NaiveBayes] Modelo cargado desde: {filepath}")
