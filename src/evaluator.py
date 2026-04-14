"""
evaluator.py
------------
Evaluación del modelo Naive Bayes.

Incluye:
  - k_folds_split : división manual del dataset en K particiones
  - compute_metrics: Precisión, Recall, F1-Score por clase,
                     Accuracy global, Macro F1 y Matriz de Confusión
  - print_report   : imprime el reporte formateado en consola
"""


def k_folds_split(data: list, k: int = 5) -> list:
    """
    Divide `data` en K particiones para cross validation.

    Parámetros
    ----------
    data : lista de elementos (tuplas documento, etiqueta)
    k    : número de folds (mínimo 5 según rúbrica)

    Retorna
    -------
    Lista de k tuplas (train_indices, test_indices)
    """
    n         = len(data)
    fold_size = n // k
    indices   = list(range(n))
    folds     = []

    for i in range(k):
        test_start = i * fold_size
        # El último fold absorbe los registros sobrantes
        test_end   = test_start + fold_size if i < k - 1 else n

        test_idx  = indices[test_start:test_end]
        train_idx = indices[:test_start] + indices[test_end:]
        folds.append((train_idx, test_idx))

    return folds


def compute_metrics(y_true: list, y_pred: list, classes: list) -> dict:
    """
    Calcula métricas de evaluación para un clasificador multiclase.

    Métricas calculadas
    -------------------
    Por clase : Precisión, Recall, F1-Score, TP, FP, FN
    Global    : Accuracy, Macro F1
    Estructura: Matriz de Confusión (filas=real, columnas=predicho)

    Fórmulas
    --------
    Precisión  = TP / (TP + FP)
    Recall     = TP / (TP + FN)
    F1-Score   = 2 * P * R / (P + R)
    Accuracy   = correctos / total
    Macro F1   = promedio de F1 por clase
    """
    # Inicializar matriz de confusión
    confusion = {cls: {c: 0 for c in classes} for cls in classes}

    for true, pred in zip(y_true, y_pred):
        confusion[true][pred] += 1

    per_class = {}
    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in classes if other != cls)
        fn = sum(confusion[cls][other] for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall) / (precision + recall) \
                    if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'tp':        tp,
            'fp':        fp,
            'fn':        fn,
        }

    accuracy  = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    macro_f1  = sum(per_class[cls]['f1'] for cls in classes) / len(classes)

    return {
        'per_class':        per_class,
        'accuracy':         accuracy,
        'macro_f1':         macro_f1,
        'confusion_matrix': confusion,
    }


def print_report(results: dict, classes: list, fold: int = None) -> None:
    """Imprime el reporte de evaluación en consola de forma legible."""
    header = f"FOLD {fold}" if fold is not None else "RESULTADO FINAL"
    print(f"\n{'='*65}")
    print(f"{header:^65}")
    print(f"{'='*65}")
    print(f"\n{'Clase':<26} {'Precisión':>10} {'Recall':>10} {'F1-Score':>10}")
    print("-" * 58)
    for cls in classes:
        m = results['per_class'][cls]
        print(f"{cls:<26} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print("-" * 58)
    print(f"\n  Accuracy Global : {results['accuracy']:.4f}")
    print(f"  Macro F1-Score  : {results['macro_f1']:.4f}")

    print(f"\n  Matriz de Confusión (filas=real, columnas=predicho):")
    col_w = 14
    print(" " * 26 + "".join(f"{c[:col_w]:>{col_w}}" for c in classes))
    for cls in classes:
        row = f"{cls[:26]:<26}" + "".join(
            f"{results['confusion_matrix'][cls][c]:>{col_w}}" for c in classes
        )
        print(row)
    print()
