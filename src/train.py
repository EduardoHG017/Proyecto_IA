"""
train.py
--------
Script principal de entrenamiento del clasificador Naive Bayes.

Flujo:
  1. Cargar y preprocesar el dataset CSV (Bitext Customer Support)
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

# Permitir imports desde src/ cuando se ejecuta desde la raiz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from preprocessor import preprocess
from naive_bayes   import NaiveBayesClassifier
from evaluator     import k_folds_split, compute_metrics, print_report


# ---------------------------------------------------------------------------
# Las categorias del dataset Bitext ya vienen en ingles y son consistentes
# con el contenido del texto, por lo que no necesitan mapeo adicional.
# Las 11 categorias son:
#   ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK,
#   INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CARGA DEL DATASET
# ---------------------------------------------------------------------------
def load_dataset(filepath: str) -> tuple:
    """
    Lee el CSV del dataset Bitext y retorna (documents, labels).

    El dataset tiene dos columnas relevantes:
      - instruction : texto de la solicitud del cliente
      - category    : etiqueta de clase (ORDER, BILLING, SHIPPING, etc.)

    Aplica preprocesamiento completo a cada solicitud.
    """
    documents = []
    labels    = []

    print(f"\nCargando dataset desde: {filepath}")
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text     = row.get('instruction', '').strip()
            category = row.get('category',    '').strip()

            # Solo procesar si tiene categoria valida y texto no vacio
            if category and text:
                tokens = preprocess(text)
                if tokens:  # ignorar documentos vacios tras preprocesamiento
                    documents.append(tokens)
                    labels.append(category)

            if (i + 1) % 5000 == 0:
                print(f"  Procesados {i + 1:,} registros...")

    print(f"  Total cargado: {len(documents):,} documentos | {len(set(labels))} clases")
    print(f"  Clases: {sorted(set(labels))}")
    return documents, labels


# ---------------------------------------------------------------------------
# K-FOLDS CROSS VALIDATION
# ---------------------------------------------------------------------------
def run_k_folds(documents: list, labels: list, k: int = 5) -> list:
    """
    Ejecuta K-Folds Cross Validation de forma manual.

    En cada iteracion:
      - Se usan K-1 folds para entrenar un nuevo modelo
      - Se usa 1 fold para evaluar
      - Se calculan las metricas del fold
    Al final se promedian los resultados de todos los folds.

    Retorna:
        Lista con los resultados de cada fold
    """
    classes = sorted(set(labels))
    data    = list(zip(documents, labels))

    # Mezcla reproducible con semilla fija
    random.seed(42)
    random.shuffle(data)

    folds       = k_folds_split(data, k=k)
    all_results = []

    print(f"\n{'='*65}")
    print(f"  Iniciando {k}-Folds Cross Validation")
    print(f"{'='*65}")

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        train_data = [data[i] for i in train_idx]
        test_data  = [data[i] for i in test_idx]

        train_docs   = [d[0] for d in train_data]
        train_labels = [d[1] for d in train_data]
        test_docs    = [d[0] for d in test_data]
        test_labels  = [d[1] for d in test_data]

        # Entrenar un nuevo clasificador con los datos de este fold
        clf = NaiveBayesClassifier()
        clf.train(train_docs, train_labels)

        # Predecir las etiquetas del conjunto de prueba
        predictions = [clf.predict(doc)[0] for doc in test_docs]

        # Calcular y mostrar metricas del fold
        results = compute_metrics(test_labels, predictions, classes)
        all_results.append(results)
        print_report(results, classes, fold=fold_idx)

    # ---- Promedios y varianza entre folds ----
    print(f"\n{'='*65}")
    print(f"{'RESUMEN PROMEDIO K-FOLDS':^65}")
    print(f"{'='*65}")

    accuracies = [r['accuracy'] for r in all_results]
    macro_f1s  = [r['macro_f1'] for r in all_results]

    avg_acc = sum(accuracies) / k
    avg_f1  = sum(macro_f1s)  / k
    std_acc = (sum((a - avg_acc) ** 2 for a in accuracies) / k) ** 0.5
    std_f1  = (sum((f - avg_f1)  ** 2 for f in macro_f1s)  / k) ** 0.5

    print(f"\n  Accuracy promedio : {avg_acc:.4f}  +/-  {std_acc:.4f}")
    print(f"  Macro F1 promedio : {avg_f1:.4f}  +/-  {std_f1:.4f}")

    print(f"\n{'Clase':<16} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 48)
    for cls in classes:
        avg_p = sum(r['per_class'][cls]['precision'] for r in all_results) / k
        avg_r = sum(r['per_class'][cls]['recall']    for r in all_results) / k
        avg_f = sum(r['per_class'][cls]['f1']        for r in all_results) / k
        print(f"{cls:<16} {avg_p:>10.4f} {avg_r:>10.4f} {avg_f:>10.4f}")

    return all_results


# ---------------------------------------------------------------------------
# ENTRENAMIENTO FINAL
# ---------------------------------------------------------------------------
def train_final_model(documents: list, labels: list, model_path: str) -> NaiveBayesClassifier:
    """
    Entrena el modelo final usando el 100% de los datos disponibles.

    Una vez validado con K-Folds, se entrena con todos los datos para
    aprovechar al maximo la informacion antes de guardar el modelo.
    """
    print(f"\n{'='*65}")
    print("  Entrenando modelo final con todos los datos...")
    print(f"{'='*65}")

    clf = NaiveBayesClassifier()
    clf.train(documents, labels)

    # Crear la carpeta model/ si no existe y guardar el modelo
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    clf.save(model_path)

    print(f"  Vocabulario: {len(clf.vocabulary):,} palabras unicas")
    print(f"  Clases: {sorted(clf.classes)}")
    return clf


# ---------------------------------------------------------------------------
# PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH  = os.path.join(BASE_DIR, 'data', 'bitext_support.csv')
    MODEL_PATH = os.path.join(BASE_DIR, 'model', 'modelo_nb.pkl')

    documents, labels = load_dataset(DATA_PATH)
    run_k_folds(documents, labels, k=5)
    train_final_model(documents, labels, MODEL_PATH)

    print("\n  Proceso de entrenamiento finalizado.")
