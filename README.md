# Clasificación de Solicitudes a Mesa de Ayuda
**Universidad Rafael Landívar · Inteligencia Artificial · Primer Semestre 2026**

Sistema de clasificación automática de tickets de soporte técnico usando **Naive Bayes Multinomial** implementado desde cero.

---

## Descripción

El sistema clasifica tickets de soporte en 5 categorías:

| Categoría | Descripción |
|---|---|
| Soporte Técnico | Problemas técnicos con productos/servicios |
| Facturación | Consultas sobre cargos y pagos |
| Consulta General | Preguntas sobre productos |
| Queja | Solicitudes de reembolso o quejas |
| Cancelación | Solicitudes de cancelación del servicio |

---

## Tecnologías

- **Backend**: Python 3.10+, Flask
- **Algoritmo**: Naive Bayes Multinomial (implementación manual)
- **Preprocesamiento**: NLTK (tokenización + stopwords + PorterStemmer)
- **Frontend**: HTML5 + CSS3 + JavaScript + Bootstrap 5
- **Dataset**: Customer Support Ticket Dataset (~8,400 registros)

---

## Estructura del Proyecto

```
Proyecto_IA/
├── data/
│   └── customer_support_tickets.csv   # Dataset de entrenamiento
├── src/
│   ├── preprocessor.py   # Limpieza, tokenización, stemming
│   ├── naive_bayes.py    # Algoritmo Naive Bayes desde cero
│   ├── evaluator.py      # K-Folds manual + métricas
│   └── train.py          # Script de entrenamiento
├── model/
│   └── modelo_nb.pkl     # Modelo entrenado (generado por train.py)
├── templates/
│   └── index.html        # Interfaz web
├── static/
│   ├── css/style.css
│   └── js/script.js
├── app.py                # Servidor Flask
├── requirements.txt
└── README.md
```

---

## Instalación y Ejecución

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo
```bash
python src/train.py
```
Esto ejecuta K-Folds Cross Validation (K=5), imprime las métricas y guarda el modelo en `model/modelo_nb.pkl`.

### 3. Iniciar la aplicación web
```bash
python app.py
```

### 4. Abrir en el navegador
```
http://localhost:5000
```

---

## Técnicas Implementadas

| Técnica | Archivo | Descripción |
|---|---|---|
| Bag of Words | `naive_bayes.py` | Vocabulario construido desde el corpus |
| Laplace Smoothing | `naive_bayes.py` | P(w\|clase) = (count+1)/(N+\|V\|) |
| Suma de Logaritmos | `naive_bayes.py` | Evita underflow numérico en inferencia |
| K-Folds (K=5) | `evaluator.py` | Implementación manual sin librerías |
| Matriz de Confusión | `evaluator.py` | 5×5 con análisis por clase |
| Métricas completas | `evaluator.py` | Precisión, Recall, F1, Accuracy, Macro F1 |

---

## Notas

- **No se usa** scikit-learn, TensorFlow, Keras, PyTorch ni ninguna librería de clasificación automática.
- NLTK se usa **únicamente** para tokenización y stopwords, no para clasificación.
- El modelo se guarda en formato **pickle** y se carga automáticamente al iniciar Flask.
