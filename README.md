# Asistente Jurídico Tributario

Aplicación de asistencia jurídica tributaria que utiliza técnicas avanzadas de Retrieval Augmented Generation (RAG) para proporcionar respuestas precisas a consultas jurídicas en el área de derecho tributario colombiano.

## Características

- **Interfaz de usuario intuitiva**: Aplicación web desarrollada con Streamlit
- **Múltiples bases de conocimiento**: 
  - Dian varios: Consultas sobre conceptos varios de la Dian
  - Renta: Consultas sobre el Impuesto de Renta
  - Renta Experimental: Versión experimental con flujos avanzados
- **Tecnologías RAG avanzadas**:
  - Utiliza LangChain y LangGraph para flujos de procesamiento
  - Integración con Chroma DB y Pinecone para almacenamiento vectorial
  - Soporte para modelos de OpenAI y Anthropic

## Estructura del Proyecto

- `Inicio.py`: Página principal de la aplicación Streamlit
- `pages/`: Directorio con páginas adicionales de la aplicación
  - `1_Dian_varios.py`: Página para consultas sobre conceptos de la Dian
  - `2_Renta.py`: Página para consultas sobre Impuesto de Renta
  - `3_Renta_Experimental.py`: Versión experimental con flujos avanzados
- `graph/`: Directorio con definiciones de grafos LangGraph
- `legal_docs/`: Documentos legales utilizados para entrenamiento
- `.chroma/`: Base de datos vectorial local (Chroma)
- `data/`: Datos adicionales para la aplicación

## Requisitos

- Python 3.10 o superior
- Poetry (gestor de dependencias)
- Claves API para:
  - OpenAI
  - Anthropic (opcional)
  - Pinecone
  - Tavily (opcional)
  - LangChain (opcional para trazabilidad)

## Instalación

1. Clona este repositorio:
   ```
   git clone https://github.com/tu-usuario/asistente-juridico-tributario.git
   cd asistente-juridico-tributario
   ```

2. Instala las dependencias con Poetry:
   ```
   poetry install
   ```

3. Crea un archivo `.env` con tus claves API (usa `.env.example` como referencia)

## Ejecución

1. Activa el entorno virtual:
   ```
   poetry shell
   ```

2. Inicia la aplicación Streamlit:
   ```
   streamlit run Inicio.py
   ```

3. Abre tu navegador en `http://localhost:8501`

## Despliegue en Streamlit Cloud

1. Crea una cuenta en [Streamlit Cloud](https://streamlit.io/cloud)
2. Conecta tu repositorio de GitHub
3. Configura las variables de entorno en la interfaz de Streamlit Cloud
4. Despliega la aplicación

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Ver el archivo `LICENSE` para más detalles.