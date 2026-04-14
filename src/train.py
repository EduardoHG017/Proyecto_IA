"""
train.py
--------
Script principal de entrenamiento.

Flujo:
  1. Cargar y preprocesar el dataset CSV
  2. Ejecutar K-Folds Cross Validation (K=5) manual
  3. Entrenar el modelo final con todos los datos
  4. Guardar el modelo en model/modelo_nb.pkl

Uso:
    python src/train.py
"""

import csv
import os
import random
import sys

# Permitir imports desde src/ cuando se ejecuta desde la raíz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from preprocessor import preprocess
from naive_bayes   import NaiveBayesClassifier
from evaluator     import k_folds_split, compute_metrics, print_report


# ---------------------------------------------------------------------------
# Mapeo de categorías originales → nombres del proyecto
# ---------------------------------------------------------------------------
CATEGORY_MAP = {
    'Technical issue':     'Soporte Técnico',
    'Billing inquiry':     'Facturación',
    'Product inquiry':     'Consulta General',
    'Refund request':      'Queja',
    'Cancellation request':'Cancelación',
}


# ---------------------------------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------------------------------
def load_dataset(filepath: str) -> tuple:
    """
    Lee el CSV y retorna (documents, labels).

    Combina 'Ticket Subject' + 'Ticket Description' como texto de entrada.
    Aplica preprocesamiento completo a cada registro.
    """
    documents = []
    labels    = []

    print(f"\nCargando dataset desde: {filepath}")
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            subject     = row.get('Ticket Subject', '')
            description = row.get('Ticket Description', '')
            text        = f"{subject} {description}"

            label_raw = row.get('Ticket Type', '').strip()
            label     = CATEGORY_MAP.get(label_raw)

            if label and text.strip():
                tokens = preprocess(text)
                if tokens:
                    documents.append(tokens)
                    labels.append(label)

            if (i + 1) % 5000 == 0:
                print(f"  Procesados {i + 1:,} registros...")

    print(f"  Total cargado: {len(documents):,} documentos | {len(set(labels))} clases")
    return documents, labels


# ---------------------------------------------------------------------------
# K-FOLDS CROSS VALIDATION
# ---------------------------------------------------------------------------
def run_k_folds(documents: list, labels: list, k: int = 5) -> list:
    """
    Ejecuta K-Folds Cross Validation de forma manual.

    Retorna la lista de resultados de cada fold.
    """
    classes = sorted(set(labels))
    data    = list(zip(documents, labels))

    # Mezcla reproducible
    random.seed(42)
    random.shuffle(data)

    folds      = k_folds_split(data, k=k)
    all_results = []

    print(f"\n{'='*65}")
    print(f"  {k}-Folds Cross Validation")
    print(f"{'='*65}")

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        train_data = [data[i] for i in train_idx]
        test_data  = [data[i] for i in test_idx]

        train_docs   = [d[0] for d in train_data]
        train_labels = [d[1] for d in train_data]
        test_docs    = [d[0] for d in test_data]
        test_labels  = [d[1] for d in test_data]

        clf = NaiveBayesClassifier()
        clf.train(train_docs, train_labels)

        predictions = [clf.predict(doc)[0] for doc in test_docs]

        results = compute_metrics(test_labels, predictions, classes)
        all_results.append(results)

        print_report(results, classes, fold=fold_idx)

    # ---- Promedios y varianza entre folds ----
    print(f"\n{'='*65}")
    print(f"{'RESUMEN K-FOLDS':^65}")
    print(f"{'='*65}")

    accuracies = [r['accuracy'] for r in all_results]
    macro_f1s  = [r['macro_f1'] for r in all_results]

    avg_acc = sum(accuracies) / k
    avg_f1  = sum(macro_f1s)  / k
    std_acc = (sum((a - avg_acc) ** 2 for a in accuracies) / k) ** 0.5
    std_f1  = (sum((f - avg_f1)  ** 2 for f in macro_f1s)  / k) ** 0.5

    print(f"\n  Accuracy promedio : {avg_acc:.4f}  ±  {std_acc:.4f}")
    print(f"  Macro F1 promedio : {avg_f1:.4f}  ±  {std_f1:.4f}")

    print(f"\n{'Clase':<26} {'Precisión':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 58)
    for cls in classes:
        avg_p = sum(r['per_class'][cls]['precision'] for r in all_results) / k
        avg_r = sum(r['per_class'][cls]['recall']    for r in all_results) / k
        avg_f = sum(r['per_class'][cls]['f1']        for r in all_results) / k
        print(f"{cls:<26} {avg_p:>10.4f} {avg_r:>10.4f} {avg_f:>10.4f}")

    return all_results


# ---------------------------------------------------------------------------
# ENTRENAMIENTO FINAL
# ---------------------------------------------------------------------------
def train_final_model(documents: list, labels: list, model_path: str) -> NaiveBayesClassifier:
    """Entrena con el 100% del dataset y guarda el modelo."""
    print(f"\n{'='*65}")
    print("  Entrenando modelo final con todos los datos...")
    print(f"{'='*65}")

    clf = NaiveBayesClassifier()
    clf.train(documents, labels)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    clf.save(model_path)
    print(f"  Vocabulario: {len(clf.vocabulary):,} palabras únicas")
    print(f"  Clases: {clf.classes}")
    return clf


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH  = os.path.join(BASE_DIR, 'data', 'customer_support_tickets.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'modelo_nb.pkl')

    documents, labels = load_dataset(DATA_PATH)
    run_k_folds(documents, labels, k=5)
    train_final_model(documents, labels, MODEL_PATH)

    print("\n  Entrenamiento completado exitosamente.")
