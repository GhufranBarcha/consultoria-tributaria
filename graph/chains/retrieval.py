import os
from typing import List
from dotenv import load_dotenv
import pinecone
from openai import OpenAI
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Cargar variables de entorno
load_dotenv()

# Configuración para Chroma (IVA)
chroma_retriever = Chroma(
    collection_name="legal-docs-chroma",
    persist_directory="./.chroma",
    embedding_function=OpenAIEmbeddings(),
).as_retriever()

# Configuración para Pinecone (Renta)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ejhr")
NAMESPACE = "renta"  # Usar el namespace "renta" para buscar documentos de renta
EMBEDDING_MODEL = "text-embedding-3-large"
TOP_K = 5  # Número de resultados a recuperar

# Inicializar cliente de OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text: str) -> List[float]:
    """
    Obtiene el embedding para un texto usando OpenAI.
    """
    response = client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def initialize_pinecone():
    """
    Inicializa la conexión con Pinecone.
    """
    try:
        print(f"initialize_pinecone: Inicializando Pinecone con API_KEY={PINECONE_API_KEY[:4]}... y ENVIRONMENT={PINECONE_ENVIRONMENT}")
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        
        # Verificar si el índice existe
        existing_indexes = [index.name for index in pc.list_indexes()]
        print(f"initialize_pinecone: Índices existentes: {existing_indexes}")
        
        if INDEX_NAME not in existing_indexes:
            print(f"initialize_pinecone: El índice {INDEX_NAME} no existe.")
            return None
        
        print(f"initialize_pinecone: Conectando al índice {INDEX_NAME}")
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error al inicializar Pinecone: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def query_pinecone(query: str, top_k: int = TOP_K):
    """
    Consulta Pinecone para obtener documentos relevantes.
    """
    print(f"query_pinecone: Consultando Pinecone para: '{query}'")
    # Inicializar Pinecone
    index = initialize_pinecone()
    if not index:
        print("query_pinecone: No se pudo inicializar Pinecone")
        return []
    
    try:
        # Obtener embedding para la consulta
        print("query_pinecone: Obteniendo embedding para la consulta")
        query_embedding = get_embedding(query)
        
        # Consultar Pinecone
        print(f"query_pinecone: Consultando índice {INDEX_NAME}, namespace '{NAMESPACE}'")
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=NAMESPACE,
            include_metadata=True
        )
        
        print(f"query_pinecone: Resultados obtenidos: {len(results.matches)}")
        
        # Imprimir información sobre los resultados
        for i, match in enumerate(results.matches):
            print(f"  Resultado {i+1}: score={match.score}, source={match.metadata.get('source', 'N/A')}")
        
        # Convertir resultados a documentos de Langchain
        documents = []
        for i, match in enumerate(results.matches):
            # Crear una fuente que sea claramente de Pinecone y no contenga "legal_docs"
            original_source = match.metadata.get('source', f'Documento-Pinecone-{i+1}')
            
            # Reemplazar cualquier referencia a legal_docs con pinecone_docs
            if 'legal_docs' in original_source:
                source = original_source.replace('legal_docs', 'pinecone_docs')
            else:
                source = f"pinecone_docs/{original_source}"
                
            doc = Document(
                page_content=match.metadata.get('text', ''),
                metadata={
                    'source': source,
                    'score': match.score,
                    'page': match.metadata.get('page', 0)
                }
            )
            documents.append(doc)
        
        print(f"query_pinecone: Documentos convertidos: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error al consultar Pinecone: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

class MultiRetriever:
    """
    Retriever que puede consultar diferentes fuentes según el tema.
    """
    def invoke(self, query: str, topic: str = None):
        """
        Invoca el retriever adecuado según el tema.
        """
        print(f"MultiRetriever: Tema seleccionado = '{topic}'")
        
        # Asegurarnos de que topic es una cadena y hacer una comparación exacta
        if topic is not None and topic.strip() == "Renta":
            # Usar Pinecone para consultas de Renta
            print("MultiRetriever: Usando Pinecone para consultas de Renta")
            docs = query_pinecone(query)
            print(f"MultiRetriever: Recuperados {len(docs)} documentos de Pinecone")
            return docs
        else:
            # Usar Chroma para otros temas (por defecto IVA)
            print(f"MultiRetriever: Usando Chroma para consultas de '{topic if topic else 'IVA'}'")
            docs = chroma_retriever.invoke(query)
            print(f"MultiRetriever: Recuperados {len(docs)} documentos de Chroma")
            return docs

# Crear una instancia del MultiRetriever
retriever = MultiRetriever() 