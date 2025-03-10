#!/usr/bin/env python
"""
Script para ingestar documentos de renta en Pinecone usando text-embedding-3-large.
"""

import os
import glob
import time
import shutil
from typing import List, Dict, Any
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredHTMLLoader,
    TextLoader,
    DirectoryLoader
)

# Cargar variables de entorno
load_dotenv()

# Configuración
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ejhr")
NAMESPACE = "renta"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSION = 3072  # Dimensión para text-embedding-3-large
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100  # Número de documentos a procesar por lote

# Inicializar cliente de OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def load_document(file_path: str) -> List[Document]:
    """
    Carga un documento según su tipo de archivo.
    """
    try:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_path.lower().endswith(".html"):
            loader = UnstructuredHTMLLoader(file_path)
            return loader.load()
        elif file_path.lower().endswith(".txt"):
            loader = TextLoader(file_path)
            return loader.load()
        else:
            print(f"Tipo de archivo no soportado: {file_path}")
            return []
    except Exception as e:
        print(f"Error al cargar el archivo {file_path}: {str(e)}")
        return []

def load_documents_from_directory(directory: str) -> List[Document]:
    """
    Carga todos los documentos de un directorio.
    """
    all_files = []
    for ext in ["pdf", "html", "txt"]:
        all_files.extend(glob.glob(f"{directory}/**/*.{ext}", recursive=True))
    
    documents = []
    errors = 0
    
    print(f"Encontrados {len(all_files)} archivos para procesar")
    
    for file_path in all_files:
        try:
            docs = load_document(file_path)
            if docs:
                documents.extend(docs)
                print(f"Cargado: {file_path} - {len(docs)} documentos")
            else:
                print(f"No se pudo cargar: {file_path}")
                errors += 1
        except Exception as e:
            print(f"Error al procesar {file_path}: {str(e)}")
            errors += 1
    
    print(f"Carga completada. {len(documents)} documentos cargados. {errors} errores.")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Divide los documentos en chunks más pequeños.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos en {len(chunks)} chunks")
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Obtiene embeddings para una lista de textos usando OpenAI.
    """
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        response = client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Procesados {len(embeddings)}/{len(texts)} embeddings")
        # Pequeña pausa para evitar límites de rate
        time.sleep(0.5)
    
    return embeddings

def initialize_pinecone():
    """
    Inicializa la conexión con Pinecone y verifica que el índice exista.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # Verificar si el índice existe
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"El índice {INDEX_NAME} no existe en Pinecone.")
        print("Por favor, crea el índice manualmente en la consola de Pinecone con las siguientes especificaciones:")
        print(f"- Nombre: {INDEX_NAME}")
        print(f"- Dimensión: {EMBEDDING_DIMENSION}")
        print("- Métrica: cosine")
        print("- Tipo: Serverless")
        return None
    
    print(f"Índice {INDEX_NAME} encontrado. Procediendo con la ingesta...")
    return pc.Index(INDEX_NAME)

def upsert_to_pinecone(index, chunks: List[Document]):
    """
    Inserta los chunks en Pinecone.
    """
    # Preparar los textos y metadatos
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Obtener embeddings
    embeddings = get_embeddings(texts)
    
    # Preparar los vectores para Pinecone
    vectors = []
    for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
        vectors.append({
            "id": f"renta-doc-{i}",
            "values": embedding,
            "metadata": {
                "text": text[:1000],  # Limitar el tamaño del texto en los metadatos
                "source": metadata.get("source", ""),
                "page": metadata.get("page", 0)
            }
        })
    
    # Insertar en lotes
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i+BATCH_SIZE]
        index.upsert(vectors=batch, namespace=NAMESPACE)
        print(f"Insertados {i+len(batch)}/{len(vectors)} vectores en Pinecone")
        time.sleep(1)  # Pequeña pausa para evitar límites de rate

def main():
    """
    Función principal para la ingesta de documentos.
    """
    # Directorio donde se encuentran los documentos de renta
    docs_dir = "data/renta"
    
    # Verificar si el directorio existe
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir, exist_ok=True)
        print(f"Directorio {docs_dir} creado. Por favor, coloca tus documentos de renta en este directorio.")
        return
    
    # Cargar documentos
    print("Cargando documentos...")
    documents = load_documents_from_directory(docs_dir)
    
    if not documents:
        print("No se encontraron documentos para procesar.")
        return
    
    # Dividir documentos en chunks
    print("Dividiendo documentos en chunks...")
    chunks = split_documents(documents)
    
    # Inicializar Pinecone
    print("Inicializando Pinecone...")
    index = initialize_pinecone()
    
    if index is None:
        return
    
    # Insertar chunks en Pinecone
    print("Insertando chunks en Pinecone...")
    upsert_to_pinecone(index, chunks)
    
    print("Ingesta completada con éxito.")

if __name__ == "__main__":
    main() 