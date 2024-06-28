# Build Your Own Search Engine

## Descripción del Taller
Este proyecto es parte de un taller donde se construye un motor de búsqueda utilizando documentos de preguntas frecuentes de varios cursos de Zoomcamp. Los resultados pueden ser utilizados posteriormente para un sistema de preguntas y respuestas.

## Objetivos del Taller
1. Preparar el entorno.
2. Aprender los conceptos básicos de búsqueda de texto e información.
3. Implementar la búsqueda de texto básica usando TF-IDF.
4. Introducir embeddings y búsqueda vectorial utilizando técnicas como Word2Vec, LSA y BERT.
5. Combinar búsqueda de texto y vectorial.
6. Explorar herramientas y aspectos prácticos de implementación en el mundo real.

## Implementación de la Búsqueda de Texto
- **Preparar el entorno**: Instalar las bibliotecas necesarias y descargar los datos.
- **Búsqueda de texto básica**:
  - Filtrado por palabras clave.
  - Vectorización con CountVectorizer y TfidfVectorizer.
  - Calcular similitudes de consulta-documento.
  - Búsqueda en todos los documentos y campos específicos.
- **Búsqueda vectorial**:
  - Utilizar técnicas de embeddings como SVD, NMF y BERT.
  - Calcular similitudes de consulta-documento con embeddings.
  - Implementar y buscar usando embeddings de BERT.

## Uso de la Clase TextSearch
Se crea una clase `TextSearch` para encapsular toda la funcionalidad de búsqueda y simplificar el uso.

## Recursos
- [Código del Taller](https://github.com/alexeygrigorev/minsearch)
- [Video del Taller](https://www.youtube.com/watch?v=nMrGK5QgPVE)
- [Registro](https://lu.ma/jsyob4df)

## Preparar el Entorno

Primero, se instalan las bibliotecas necesarias

```bash
$ pip install requests pandas scikit-learn jupyter
```

Levantamos el servidor de Jupyter Notebooks:

```bash
$ jupyter notebook
```

Se descarga el conjunto de datos de documentos:
```python
import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

Aquí se crea un DataFrame de pandas a partir de los datos descargados:

```python
import pandas as pd

df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])
df.head()
```

## Búsqueda de Texto Básica

### Filtrado por Palabras Clave

```python
df[df.course == 'data-engineering-zoomcamp'].head()
```

### Vectorización y Similitud

La vectorización es el proceso de convertir texto en una representación numérica, es decir, en vectores, que se pueden usar para cálculos matemáticos como la similitud. En este apartado, usaremos dos métodos de vectorización: CountVectorizer y TfidfVectorizer.
CountVectorizer convierte una colección de documentos de texto en una matriz de tokens de recuento. En otras palabras, cuenta el número de veces que aparece cada palabra en cada documento.

Se muestra cómo usar CountVectorizer y TfidfVectorizer para vectorizar el texto y calcular similitudes entre consultas y documentos:

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "Course starts on 15th Jan 2024",
    "Prerequisites listed on GitHub",
    "Submit homeworks after start date",
    "Registration not required for participation",
    "Setup Google Cloud and Python before course"
]

# Crear el Vectorizador:
cv = CountVectorizer(stop_words='english')

# Ajustar y Transformar los Documentos:
X = cv.fit_transform(documents)

tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(documents)
```

Para una consulta, se transforma y se calcula la similitud:

```python
query = "Do I need to know python to sign up for the January course?"

q = tfidf.transform([query])
score = cosine_similarity(X_tfidf, q).flatten()
```

## Búsqueda Vectorial con Embeddings

### Embeddings con SVD y NMF

Se usa TruncatedSVD y NMF para reducir la dimensionalidad y calcular similitudes:

```python
from sklearn.decomposition import TruncatedSVD, NMF

svd = TruncatedSVD(n_components=16)
X_svd = svd.fit_transform(X_tfidf)

nmf = NMF(n_components=16)
X_nmf = nmf.fit_transform(X_tfidf)
```

### Embeddings con BERT

Se utiliza la biblioteca transformers de Hugging Face para crear embeddings con BERT:

```python
pip install transformers tqdm

import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

texts = df['text'].tolist()
text_batches = make_batches(texts, 8)

all_embeddings = []

for batch in tqdm(text_batches):
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state
        batch_embeddings = hidden_states.mean(dim=1)
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings_np)

final_embeddings = np.vstack(all_embeddings)
```

### Clase TextSearch

Se crea una clase para encapsular la funcionalidad de búsqueda:

```python
class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
```

La clase se utiliza de la siguiente manera:

```python
index = TextSearch(
    text_fields=['section', 'question', 'text']
)
index.fit(documents)

index.search(
    query='I just singned up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)
```