from dotenv import load_dotenv
import os
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredHTMLLoader, TextLoader, DirectoryLoader
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Ruta a la carpeta que contiene tus documentos jurídicos
DOCS_DIR = "legal_docs"  # Cambia esto a la ruta de tu carpeta

# Primero, eliminar la base de datos Chroma existente
CHROMA_DIR = "./.chroma"
if os.path.exists(CHROMA_DIR):
    print(f"Eliminando base de datos Chroma existente en {CHROMA_DIR}...")
    shutil.rmtree(CHROMA_DIR)
    print("Base de datos eliminada.")

# Verificar si el directorio existe
if not os.path.exists(DOCS_DIR):
    print(f"El directorio {DOCS_DIR} no existe. Creándolo...")
    os.makedirs(DOCS_DIR)
    print(f"Por favor, coloca tus documentos PDF y HTML en la carpeta {DOCS_DIR} y ejecuta este script nuevamente.")
    exit()

# Cargar documentos
documents = []
print(f"Cargando documentos desde {DOCS_DIR}...")

# Función para cargar documentos con manejo de errores
def load_document(file_path, loader_class, file_type):
    try:
        print(f"Cargando {file_type}: {os.path.basename(file_path)}")
        loader = loader_class(file_path)
        return loader.load()
    except Exception as e:
        print(f"Error al cargar {file_path}: {str(e)}")
        return []

# Contar archivos
file_count = 0
error_count = 0

for filename in os.listdir(DOCS_DIR):
    file_path = os.path.join(DOCS_DIR, filename)
    
    # Ignorar directorios y archivos ocultos
    if os.path.isdir(file_path) or filename.startswith('.'):
        continue
    
    # Cargar PDFs
    if filename.lower().endswith('.pdf'):
        docs = load_document(file_path, PyPDFLoader, "PDF")
        if docs:
            documents.extend(docs)
            file_count += 1
        else:
            error_count += 1
    
    # Cargar HTMLs
    elif filename.lower().endswith(('.html', '.htm')):
        try:
            docs = load_document(file_path, UnstructuredHTMLLoader, "HTML")
            if docs:
                documents.extend(docs)
                file_count += 1
            else:
                error_count += 1
        except ImportError:
            # Alternativa si unstructured falla
            print(f"Intentando cargar {filename} como archivo de texto...")
            docs = load_document(file_path, TextLoader, "Texto")
            if docs:
                documents.extend(docs)
                file_count += 1
            else:
                error_count += 1
    
    # Cargar archivos de texto
    elif filename.lower().endswith(('.txt', '.md', '.csv')):
        docs = load_document(file_path, TextLoader, "Texto")
        if docs:
            documents.extend(docs)
            file_count += 1
        else:
            error_count += 1
    else:
        print(f"Tipo de archivo no soportado: {filename}")

if file_count == 0:
    print(f"No se pudieron cargar archivos desde {DOCS_DIR}. Por favor, verifica los formatos soportados (PDF, HTML, TXT).")
    exit()

print(f"Se cargaron {len(documents)} documentos de {file_count} archivos. Hubo {error_count} errores.")

# Dividir documentos en chunks
print("Dividiendo documentos en chunks...")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500,  # Chunks más grandes para documentos jurídicos
    chunk_overlap=50  # Algo de superposición para mantener contexto
)
doc_splits = text_splitter.split_documents(documents)
print(f"Se crearon {len(doc_splits)} chunks.")

# Crear vectorstore
print("Creando vectorstore con embeddings...")
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="legal-docs-chroma",
    embedding=OpenAIEmbeddings(),
    persist_directory=CHROMA_DIR,
)
print(f"Vectorstore creado y guardado en {CHROMA_DIR}")

# Verificar que el retriever funciona
retriever = Chroma(
    collection_name="legal-docs-chroma",
    persist_directory=CHROMA_DIR,
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

print("¡Ingesta completada con éxito!")
print(f"Número total de documentos en la base de datos: {len(doc_splits)}") 