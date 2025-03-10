# Implementación de Pinecone con text-embedding-3-large

Este proyecto implementa un sistema de búsqueda semántica para documentos de renta utilizando Pinecone como base de datos vectorial y el modelo text-embedding-3-large de OpenAI para generar embeddings.

## Requisitos previos

1. Cuenta en [Pinecone](https://www.pinecone.io/)
2. Cuenta en [OpenAI](https://openai.com/)
3. Python 3.8 o superior
4. Poetry (gestor de dependencias)

## Configuración

### 1. Crear una cuenta en Pinecone

Si aún no tienes una cuenta en Pinecone, regístrate en [https://www.pinecone.io/](https://www.pinecone.io/).

### 2. Crear un índice en Pinecone

1. Inicia sesión en la consola de Pinecone
2. Crea un nuevo índice con las siguientes especificaciones:
   - Nombre: `renta-docs` (o el nombre que prefieras, pero deberás actualizar los scripts)
   - Dimensión: 3072 (para text-embedding-3-large)
   - Métrica: cosine
   - Tipo: Serverless (recomendado para empezar)
   - Región: la más cercana a tu ubicación

### 3. Configurar variables de entorno

Edita el archivo `.env` y actualiza las siguientes variables:

```
OPENAI_API_KEY=tu-clave-de-api-de-openai
PINECONE_API_KEY=tu-clave-de-api-de-pinecone
PINECONE_ENVIRONMENT=tu-entorno-de-pinecone (por ejemplo, gcp-starter)
```

## Estructura de directorios

```
.
├── data/
│   └── renta/  # Coloca aquí tus documentos de renta (PDF, HTML, TXT)
├── ingest_renta_docs.py  # Script para ingestar documentos en Pinecone
├── query_renta_docs.py   # Script para consultar documentos desde Pinecone
└── .env                  # Archivo de variables de entorno
```

## Uso

### 1. Preparar documentos

Coloca tus documentos de renta (PDF, HTML, TXT) en el directorio `data/renta/`.

### 2. Ingestar documentos

Ejecuta el script de ingesta:

```bash
python ingest_renta_docs.py
```

Este script:
- Cargará los documentos del directorio `data/renta/`
- Los dividirá en chunks más pequeños
- Generará embeddings usando text-embedding-3-large
- Insertará los embeddings en el índice de Pinecone

### 3. Consultar documentos

Ejecuta el script de consulta:

```bash
python query_renta_docs.py
```

Este script:
- Te pedirá que ingreses una consulta sobre renta
- Buscará documentos relevantes en Pinecone
- Generará una respuesta basada en los documentos encontrados
- Mostrará la respuesta con citas a los documentos

## Personalización

Puedes modificar los siguientes parámetros en los scripts:

- `INDEX_NAME`: Nombre del índice en Pinecone
- `NAMESPACE`: Namespace dentro del índice
- `CHUNK_SIZE`: Tamaño de los chunks de texto
- `CHUNK_OVERLAP`: Superposición entre chunks
- `TOP_K`: Número de resultados a recuperar en las consultas

## Solución de problemas

### Error al crear el índice

Si encuentras errores al crear el índice, intenta crearlo manualmente desde la consola de Pinecone.

### Límites de rate en OpenAI

Si encuentras errores de límite de rate en OpenAI, puedes aumentar los tiempos de espera en las funciones `get_embeddings` y `get_embedding`.

### Documentos no encontrados

Asegúrate de que los documentos estén en el formato correcto (PDF, HTML, TXT) y en el directorio `data/renta/`.

## Recursos adicionales

- [Documentación de Pinecone](https://docs.pinecone.io/)
- [Documentación de OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [Documentación de text-embedding-3-large](https://platform.openai.com/docs/models/embeddings) 