#!/usr/bin/env python
"""
Script para consultar documentos de renta desde Pinecone usando text-embedding-3-large.
"""

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import pinecone
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configuración
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "ejhr")
NAMESPACE = "renta"
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
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    
    # Verificar si el índice existe
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"El índice {INDEX_NAME} no existe. Por favor, ejecuta primero el script de ingesta.")
        return None
    
    return pc.Index(INDEX_NAME)

def query_pinecone(index, query: str, top_k: int = TOP_K):
    """
    Consulta Pinecone para obtener documentos relevantes.
    """
    # Obtener embedding para la consulta
    query_embedding = get_embedding(query)
    
    # Consultar Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True
    )
    
    return results

def generate_response(query: str, results):
    """
    Genera una respuesta usando OpenAI basada en los resultados de Pinecone.
    """
    # Preparar el contexto con los documentos recuperados
    context = ""
    for i, match in enumerate(results.matches):
        context += f"\n\nDOCUMENTO [{i+1}]: {match.metadata.get('source', 'Desconocido')}\n"
        context += f"{match.metadata.get('text', '')}\n"
        context += f"Relevancia: {match.score:.4f}\n"
        context += "-" * 50
    
    # Crear el sistema y el mensaje del usuario
    system_message = """Eres un asistente jurídico experto especializado en derecho tributario colombiano, específicamente en temas de renta. Tu objetivo es proporcionar respuestas precisas, detalladas y fundamentadas a consultas legales.

INSTRUCCIONES PARA RESPONDER:
1. Analiza cuidadosamente la pregunta y los documentos proporcionados.
2. Identifica los aspectos legales relevantes y la normativa aplicable.
3. Estructura tu respuesta de manera clara y lógica.
4. Utiliza lenguaje técnico-jurídico preciso pero comprensible.
5. Incluye referencias específicas a artículos, normas o jurisprudencia cuando sea relevante.
6. Sé exhaustivo y detallado en tu análisis.

INSTRUCCIONES SOBRE CITAS:
1. Usa el formato de cita [n] después de cada afirmación basada en los documentos.
2. Numera las citas secuencialmente: [1], [2], [3], etc.
3. Cada número debe corresponder al documento del que extraes la información.
4. Si usas información de varios documentos en una misma afirmación, incluye todas las citas relevantes: [1][2].
5. Coloca las citas inmediatamente después de la afirmación que respaldan.
6. CADA afirmación importante debe tener su correspondiente cita.

Ejemplo de formato correcto:
"La renta es un impuesto directo que grava los ingresos de las personas [1]. La tarifa general para personas naturales varía según el nivel de ingresos [2]."

NO uses notas al pie ni referencias al final. Las citas deben estar integradas en el texto."""
    
    user_message = f"""Pregunta sobre renta: {query}

DOCUMENTOS PARA CONSULTA:
{context}

IMPORTANTE: 
1. Responde de manera completa y detallada, como un experto en derecho tributario.
2. Usa el formato de citas numéricas [1], [2], etc. después de cada afirmación que hagas.
3. Cada número debe corresponder al documento del que extraes la información.
4. Asegúrate de que CADA afirmación importante tenga su correspondiente cita entre corchetes.
5. Estructura tu respuesta con una introducción, desarrollo y conclusión."""
    
    # Llamar a la API de OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content

def main():
    """
    Función principal para consultar documentos.
    """
    # Inicializar Pinecone
    index = initialize_pinecone()
    
    if not index:
        return
    
    print("=== Consulta de Documentos de Renta ===")
    print("Escribe 'salir' para terminar")
    print()
    
    while True:
        query = input("Ingresa tu consulta sobre renta: ")
        
        if query.lower() == "salir":
            break
        
        # Consultar Pinecone
        print("\nBuscando documentos relevantes...")
        results = query_pinecone(index, query)
        
        if not results.matches:
            print("No se encontraron documentos relevantes.")
            continue
        
        # Mostrar resultados
        print(f"\nSe encontraron {len(results.matches)} documentos relevantes:")
        for i, match in enumerate(results.matches):
            print(f"{i+1}. {match.metadata.get('source', 'Desconocido')} - Relevancia: {match.score:.4f}")
        
        # Generar respuesta
        print("\nGenerando respuesta...")
        response = generate_response(query, results)
        
        print("\n=== Respuesta ===")
        print(response)
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    main() 