# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── face_pipeline.py     # Detección, alineación y preprocesamiento (compartido con las notebooks)
│       ├── schemas.py
│       ├── models/
│       │   └── custom_cnn.py    # Arquitectura de la CNN propia (modelo A)
│       ├── services/
│       │   ├── face_service.py  # detect_faces, align_face, extract_embedding_from_face
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── splits/
│   └── splits.csv               # Split de la notebook 1, reutilizado en la notebook 2
├── requirements.txt             # Dependencias de la app (Python 3.12, Docker)
├── requirements-notebooks.txt   # Dependencias con las que se ejecutaron las notebooks
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`. Si su detector viene de una librería, sobrescriban `_load_detector` en `face_service.py`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado por el grupo (`src/lib/services/face_service.py`)

Las tres funciones usan `src/lib/face_pipeline.py`, el mismo módulo que usan las notebooks, así la app procesa las caras exactamente igual que en la evaluación.

1. `detect_faces`: detecta los rostros con **SCRFD** (`det_10g.onnx`, InsightFace) y devuelve las cajas `(x1, y1, x2, y2)`. Usa `detect_with_border`: si no encuentra rostro, reintenta agregando un borde negro a la imagen, porque con fotos recortadas muy justas (como las de Pins) el detector no encuentra la cara. Descarta las cajas con un lado menor a `MIN_FACE_SIZE`.
2. `align_face`: recorta la caja con un margen (`ALIGN_MARGIN`), vuelve a detectar en el recorte para obtener los 5 landmarks y alinea a la plantilla de ArcFace en `FACE_SIZE` (112×112). Si en el recorte aparece más de una cara, usa la que más se superpone con la caja original. Si no encuentra landmarks, usa el recorte sin alinear.
3. `extract_embedding_from_face`: pasa la cara a RGB y la normaliza a `(x - 127.5) / 127.5`, corre el modelo (`.onnx` o `.pth`) y devuelve el embedding de dimensión `EMBEDDING_DIM` (512) normalizado L2.

Además, `_load_detector` carga el detector SCRFD desde `DETECTOR_MODEL_NAME`. `/insert` exige que la foto tenga exactamente una cara. `/predict` identifica cada cara que encuentra en la foto.

Ver detalles y recomendaciones en [ENUNCIADO.md](ENUNCIADO.md#4-parte-3--integración-en-la-aplicación).

## Modelo y fine-tuning

El detalle completo, con gráficos, está en [2-train.ipynb](2-train.ipynb).

### Modelo elegido: ArcFace (`w600k_r50.onnx`)

Es el reconocedor del paquete `buffalo_l` de InsightFace, preentrenado y **sin reentrenar**:

- **Arquitectura:** ResNet-50 entrenada con la loss ArcFace (margen angular aditivo) sobre WebFace600K, unas 600 mil identidades.
- **Entrada:** una cara alineada de 112×112 en RGB, normalizada a [-1, 1].
- **Salida:** un embedding de 512.
- **Tamaño:** 43.6 millones de parámetros, 174.4 MB.
- **Detector:** `det_10g.onnx` (SCRFD), del mismo paquete.
- **Descarga (link oficial):** https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip. Hay que descomprimirlo y copiar `w600k_r50.onnx` y `det_10g.onnx` a `models/`. No lo redistribuimos desde otro link por su licencia.
- **Licencia:** según el README de InsightFace, los modelos preentrenados son solo para investigación no comercial.

### Modelo descartado: CNN propia (`custom_cnn.onnx`)

Es una CNN entrenada desde cero con nuestro dataset. El código está en [src/lib/models/custom_cnn.py](src/lib/models/custom_cnn.py) y el modelo se descarga de https://drive.google.com/file/d/195_t7-_7z0snfGOiDSyX7PpTPVIg8ZXB/view?usp=sharing.

- **Arquitectura:** 5 bloques (Conv 3×3 → BatchNorm → ReLU → MaxPool) con 32 a 512 canales, después global average pooling, dropout 0.3 y una capa lineal que da el embedding de 512.
- **Tamaño:** 1.8 millones de parámetros, 7.3 MB.
- **Entrenamiento:** la red se entrena como clasificador de las 75 personas de `train`. Después se descarta el clasificador y se usa como embedding la capa anterior.

| Hiperparámetro | Valor |
|---|---|
| Optimizador | Adam, learning rate 1e-3, weight decay 5e-4 |
| Scheduler | ReduceLROnPlateau (factor 0.5, paciencia 3) |
| Batch size | 128 |
| Épocas | hasta 60, con early stopping (paciencia 8). La mejor fue la 57 |
| Loss | cross-entropy con class weights |
| Regularización | dropout 0.3, weight decay y data augmentation (solo en train) |

Probamos dos experimentos de 30 épocas:

- **Sin augmentation:** hubo overfitting. Train llegó al 100% y val se quedó en 90%.
- **Learning rate 3e-3:** el entrenamiento fue inestable y peor, con una loss de val de 0.598 contra 0.376.

Con estos resultados nos quedamos con la configuración principal.

### Protocolo de evaluación

Separamos las 105 identidades del dataset en tres grupos:

- **75 conocidas:** sus fotos se reparten en train, val y test, y sirven para entrenar la CNN.
- **15 nuevas para calibrar el umbral** (`val_nuevas`).
- **15 nuevas para reportar** (`test_nuevas`).

Salvo el accuracy closed-set, todas las métricas se miden sobre personas que ningún modelo usó para entrenar ni para elegir el umbral.

### Métricas (`test_nuevas`)

| Métrica | CNN propia | ArcFace |
|---|---|---|
| Accuracy closed-set (`test`, 75 personas conocidas) | 92.8% | — |
| AUC | 0.901 | 1.000 |
| EER | 18.5% | 0.17% |
| TAR@FAR=1e-3 | 13.4% | 99.8% |
| Top-1 / Top-5 (galería de 10 personas) | 77.2% / 98.6% | 100% / 100% |
| Precision / Recall / F1 macro | 0.843 / 0.423 / 0.486 | 1.000 / 1.000 / 1.000 |
| Desconocidos rechazados | 95.4% | 100% |
| Registrados reconocidos | 37.4% | 100% |
| ARI / NMI (K-Means) | 0.684 / 0.761 | 1.000 / 1.000 |
| Tiempo por cara (CPU) | 1.5 ms | 74.5 ms |

### Justificación y trade-offs

Elegimos ArcFace porque es muy superior en todas las métricas sobre personas nuevas, que es el uso real de la app. La CNN reconoce bien a las 75 personas con las que se entrenó (92.8%). Pero con personas nuevas y un umbral seguro reconoce solo al 37% de las registradas, así que en la práctica la app casi siempre respondería `unknown`.

La CNN tiene a favor que es 24 veces más chica, unas 50 veces más rápida y tiene licencia propia. Igual, los 75 ms por cara de ArcFace en CPU son aceptables para la app.

La CNN rinde menos por varias razones:

- Se entrenó con solo 75 identidades, contra unas 600 mil de ArcFace.
- La loss softmax no fuerza márgenes entre personas, a diferencia de ArcFace.
- Se guía mucho por la pose, la luz y el encuadre.

**Limitación:** WebFace también está armado con fotos de famosos, así que ArcFace pudo haber visto a algunas de nuestras "personas nuevas". Sus resultados pueden ser optimistas.

### Umbral de similitud: `SIMILARITY_THRESHOLD=0.27`

Lo calibramos con los pares de `val_nuevas`. Elegimos el umbral más bajo con FAR ≤ 1e-3, es decir, que acepta por error a otra persona como máximo 1 de cada 1000 veces. Dio 0.268, que redondeamos a 0.27.

Preferimos ese criterio al del EER porque confundir a alguien con otra persona registrada es el error grave, mientras que un rechazo se puede reintentar con otra foto.

En `test_nuevas`, ese umbral da una FAR de 0.02% y una FRR de 0.34%. El valor que traía la plantilla (0.55) rechazaría a más de un tercio de las personas registradas, por eso lo cambiamos en `.env.docker.example` y `.env.local.example`.

### Análisis de errores

- **ArcFace, falsos positivos:** son pocos, con similitudes de 0.28 y 0.29, apenas por encima del umbral. Son casos de parecido real, por ejemplo Mark Ruffalo con Chris Pratt.
- **ArcFace, falsos negativos:** son fotos con maquillaje de personaje muy marcado, sobreexposición u oclusiones (anteojos, una mano). Tienen similitudes de 0.12 a 0.15.
- **CNN, falsos positivos:** son pares de personas distintas con similitud de 0.68 a 0.73, casi siempre fotos con pose, encuadre y luz parecidos.
- **CNN, falsos negativos:** son pares de la misma persona con similitud negativa. Aparecen texto sobre la cara, épocas o peinados muy distintos, maquillaje de personaje y luz de color.

## Dataset

El detalle completo, con gráficos, está en [1-pre-processing.ipynb](1-pre-processing.ipynb).

- **Fuente:** [Pins Face Recognition](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition) (Kaggle, versión 1). Son imágenes públicas de Pinterest de 105 celebridades, 17534 imágenes en total, todas `.jpg` y RGB. Se descarga con `kagglehub` y no se sube al repositorio.
- **Licencia:** CC0: Public Domain, según la página del dataset. La declara quien lo armó, y los derechos de cada foto siguen siendo de sus autores. Por eso lo usamos solo con fines académicos y sin redistribuir las imágenes.
- **Cantidad por persona:** entre 86 y 237 imágenes (media 167, mediana 169).
- **Balance:** el desbalance es moderado, porque la persona con más fotos tiene unas 2.8 veces más que la que menos tiene. No filtramos identidades. Lo compensamos con class weights en la loss de la CNN.
- **Variaciones:** pose (incluidos perfiles), iluminación (escenas oscuras, luz de color, blanco y negro), expresión, edad, maquillaje y caracterizaciones de personajes, anteojos, cambios de peinado y oclusiones. Las imágenes vienen recortadas muy justas alrededor de la cara, y el 45% de los rostros mide menos de 112 px.
- **Sesgos:** son solo celebridades, mayormente de Hollywood, jóvenes y de piel clara, en fotos con buena iluminación. Las métricas pueden ser optimistas para un uso real.

### Reglas de filtrado y calidad

| Regla | Criterio | Resultado |
|---|---|---|
| Sin rostro | SCRFD no detecta ninguna cara, con reintento usando un borde negro | 0 imágenes |
| Rostro chico | lado de la cara menor a 40 px | 0 imágenes |
| Más de un rostro | nos quedamos con la cara más grande, que es la persona etiquetada | 91 imágenes, se conservan |
| Borrosas | varianza del Laplaciano menor a 50, medida sobre 112×112 | 29 descartadas |
| Exposición extrema | brillo menor a 30 o mayor a 210, o contraste menor a 20 | 42 imágenes, se conservan (son condiciones reales) |
| Duplicados y casi-duplicados | similitud coseno de los embeddings de ArcFace mayor a 0.95 | 587 descartadas |
| Outliers o posibles errores de etiqueta | similitud con el centro de su identidad menor a 0.3 | 11 descartadas |

El dataset final tiene **16907 imágenes de 105 identidades**. Verificamos que no hay casi-duplicados entre splits, para evitar fuga de datos.

| Split | Imágenes | Identidades | Uso |
|---|---|---|---|
| train / val / test | 9625 / 1172 / 1270 | 75 | Entrenar la CNN, elegir la época y medir el accuracy closed-set |
| val_nuevas | 2418 | 15 | Calibrar el umbral |
| test_nuevas | 2422 | 15 | Métricas finales open-set |

El split queda guardado en [splits/splits.csv](splits/splits.csv). Para que sea reproducible, fijamos la semilla 42 y verificamos los modelos con su hash SHA256.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
