#!/usr/bin/env python
"""
Script para probar si el flujo RAG avanzado se está utilizando correctamente en las pestañas "Dian Varios" y "Renta".
Este script simula exactamente lo que hacen estas pestañas y verifica si están utilizando el flujo RAG avanzado completo.
"""

import time
import argparse
from dotenv import load_dotenv
import traceback
import sys
import os

# Importar el grafo y componentes
from graph.graph import app, set_debug
from graph.state import GraphState
from graph.chains.retrieval import chroma_retriever, query_pinecone
from graph.chains.openai_generation import generate_with_openai

# Cargar variables de entorno
load_dotenv()

# Activar el modo de depuración para ver todos los pasos
set_debug(True)

def test_direct_approach(question: str, topic: str):
    """
    Prueba el enfoque directo (sin usar el flujo RAG avanzado).
    Este es el método que se usaba originalmente en las pestañas.
    
    Args:
        question: La consulta a procesar
        topic: El tema de la consulta (IVA o Renta)
    
    Returns:
        El resultado del enfoque directo
    """
    print(f"\n{'=' * 80}")
    print(f"PROBANDO ENFOQUE DIRECTO CON: '{question}'")
    print(f"TEMA: {topic}")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    
    try:
        # Recuperar documentos según el tema
        if topic == "Renta":
            print("Consultando directamente a Pinecone")
            documents = query_pinecone(question)
            print(f"Recuperados {len(documents)} documentos de Pinecone")
        else:
            print("Consultando directamente a Chroma")
            documents = chroma_retriever.invoke(question)
            print(f"Recuperados {len(documents)} documentos de Chroma")
        
        # Generar respuesta con OpenAI
        if documents:
            print("Generando respuesta con OpenAI")
            openai_response = generate_with_openai(question, documents)
            response = openai_response["text"]
            citations = openai_response.get("citations", [])
            print(f"Respuesta generada con {len(citations)} citas")
        else:
            response = "No se encontraron documentos relevantes"
            citations = []
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nRespuesta generada en {duration:.2f} segundos")
        
        # Mostrar un extracto de la respuesta
        print(f"\n{'=' * 40}")
        print("EXTRACTO DE LA RESPUESTA:")
        print(f"{'=' * 40}")
        print(response[:500] + "..." if len(response) > 500 else response)
        
        # Mostrar las citas
        if citations:
            print(f"\n{'=' * 40}")
            print(f"CITAS ({len(citations)}):")
            print(f"{'=' * 40}")
            for i, citation in enumerate(citations):
                print(f"\n[{i+1}] {citation.get('document_title', 'Sin título')}")
                print(f"Texto citado: \"{citation.get('cited_text', 'No disponible')}\"")
        
        return {
            "documents": documents,
            "generation": response,
            "citations": citations
        }
    
    except Exception as e:
        print(f"Error en el enfoque directo: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

def test_rag_flow(question: str, topic: str):
    """
    Prueba el flujo RAG avanzado con una consulta específica.
    Este es el método que deberían usar las pestañas después de la modificación.
    
    Args:
        question: La consulta a procesar
        topic: El tema de la consulta (IVA o Renta)
    
    Returns:
        El resultado del flujo
    """
    print(f"\n{'=' * 80}")
    print(f"PROBANDO FLUJO RAG AVANZADO CON: '{question}'")
    print(f"TEMA: {topic}")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    
    # Crear el estado inicial
    initial_state = {"question": question}
    if topic:
        initial_state["topic"] = topic
    
    print(f"\nEstado inicial: {initial_state}")
    
    try:
        # Invocar el grafo
        result = app.invoke(initial_state)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nRespuesta generada en {duration:.2f} segundos")
        
        # Mostrar el flujo completo
        print(f"\n{'=' * 40}")
        print("FLUJO COMPLETO:")
        print(f"{'=' * 40}")
        
        # Mostrar el estado final
        print(f"\nEstado final: {result.keys()}")
        
        # Mostrar documentos recuperados
        if "documents" in result:
            print(f"\n{'=' * 40}")
            print(f"DOCUMENTOS RECUPERADOS ({len(result['documents'])})")
            print(f"{'=' * 40}")
            for i, doc in enumerate(result["documents"][:3]):  # Mostrar solo los primeros 3
                print(f"\nDocumento {i+1}: {doc.metadata.get('source', 'Desconocido')}")
                print(f"{doc.page_content[:300]}...")  # Mostrar solo los primeros 300 caracteres
        
        # Mostrar la respuesta generada
        print(f"\n{'=' * 40}")
        print("EXTRACTO DE LA RESPUESTA GENERADA:")
        print(f"{'=' * 40}")
        generation = result.get("generation", "No se generó respuesta")
        print(generation[:500] + "..." if len(generation) > 500 else generation)
        
        # Mostrar las citas si existen
        if "citations" in result and result["citations"]:
            print(f"\n{'=' * 40}")
            print(f"CITAS ({len(result['citations'])})")
            print(f"{'=' * 40}")
            for i, citation in enumerate(result["citations"]):
                print(f"\n[{i+1}] {citation.get('document_title', 'Sin título')}")
                print(f"Texto citado: \"{citation.get('cited_text', 'No disponible')}\"")
        
        return result
    
    except Exception as e:
        print(f"Error en el flujo RAG avanzado: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

def compare_approaches(question: str, topic: str):
    """
    Compara el enfoque directo con el flujo RAG avanzado.
    
    Args:
        question: La consulta a procesar
        topic: El tema de la consulta (IVA o Renta)
    """
    print(f"\n{'#' * 100}")
    print(f"COMPARANDO ENFOQUES PARA: '{question}' (TEMA: {topic})")
    print(f"{'#' * 100}")
    
    # Probar el enfoque directo
    direct_result = test_direct_approach(question, topic)
    
    # Probar el flujo RAG avanzado
    rag_result = test_rag_flow(question, topic)
    
    # Comparar resultados
    print(f"\n{'#' * 100}")
    print("COMPARACIÓN DE RESULTADOS:")
    print(f"{'#' * 100}")
    
    # Comparar documentos recuperados
    direct_docs = len(direct_result.get("documents", []))
    rag_docs = len(rag_result.get("documents", []))
    print(f"\nDocumentos recuperados:")
    print(f"- Enfoque directo: {direct_docs}")
    print(f"- Flujo RAG avanzado: {rag_docs}")
    
    # Comparar longitud de respuesta
    direct_response_len = len(direct_result.get("generation", ""))
    rag_response_len = len(rag_result.get("generation", ""))
    print(f"\nLongitud de respuesta:")
    print(f"- Enfoque directo: {direct_response_len} caracteres")
    print(f"- Flujo RAG avanzado: {rag_response_len} caracteres")
    
    # Comparar citas
    direct_citations = len(direct_result.get("citations", []))
    rag_citations = len(rag_result.get("citations", []))
    print(f"\nNúmero de citas:")
    print(f"- Enfoque directo: {direct_citations}")
    print(f"- Flujo RAG avanzado: {rag_citations}")
    
    # Verificar si se está utilizando el flujo RAG avanzado
    is_using_rag = "web_search" in rag_result
    print(f"\n{'=' * 80}")
    if is_using_rag:
        print("✅ CONFIRMADO: Se está utilizando el flujo RAG avanzado completo")
        print("   - Se encontró el campo 'web_search' en el resultado, que solo se añade en el flujo avanzado")
    else:
        print("❌ ALERTA: No se está utilizando el flujo RAG avanzado completo")
        print("   - No se encontró el campo 'web_search' en el resultado")
        print("   - Es posible que solo se esté utilizando parte del flujo")
    
    # Verificar si las citas se están incluyendo correctamente
    print(f"\n{'=' * 80}")
    if "citations" in rag_result and rag_result["citations"]:
        print("✅ CONFIRMADO: Las citas se están incluyendo correctamente en el resultado del flujo RAG")
        print(f"   - Se encontraron {rag_citations} citas en el resultado")
    else:
        print("❌ ALERTA: Las citas NO se están incluyendo correctamente en el resultado del flujo RAG")
        print("   - No se encontraron citas en el resultado")
        print("   - Es posible que haya un problema en la transferencia de citas entre nodos")
    
    # Verificar si las respuestas son similares en estructura
    print(f"\n{'=' * 80}")
    direct_has_structure = "REFERENCIA" in direct_result.get("generation", "") and "ANÁLISIS" in direct_result.get("generation", "")
    rag_has_structure = "REFERENCIA" in rag_result.get("generation", "") and "ANÁLISIS" in rag_result.get("generation", "")
    
    if direct_has_structure and rag_has_structure:
        print("✅ CONFIRMADO: Ambos enfoques generan respuestas con la estructura esperada")
    elif direct_has_structure and not rag_has_structure:
        print("❌ ALERTA: Solo el enfoque directo genera respuestas con la estructura esperada")
        print("   - El flujo RAG no está utilizando el mismo formato de respuesta")
    elif not direct_has_structure and rag_has_structure:
        print("⚠️ ADVERTENCIA: Solo el flujo RAG genera respuestas con la estructura esperada")
    else:
        print("❌ ALERTA: Ninguno de los enfoques genera respuestas con la estructura esperada")
    
    print(f"{'=' * 80}")

def main():
    parser = argparse.ArgumentParser(description="Prueba el flujo RAG avanzado")
    parser.add_argument("--question", "-q", type=str, 
                        default="¿Cuáles son los requisitos para solicitar la devolución de IVA en Colombia?",
                        help="Consulta a procesar")
    parser.add_argument("--topic", "-t", type=str, default="IVA", 
                        choices=["IVA", "Renta"],
                        help="Tema de la consulta (IVA o Renta)")
    parser.add_argument("--mode", "-m", type=str, default="compare",
                        choices=["direct", "rag", "compare"],
                        help="Modo de prueba (direct, rag o compare)")
    
    args = parser.parse_args()
    
    if args.mode == "direct":
        test_direct_approach(args.question, args.topic)
    elif args.mode == "rag":
        test_rag_flow(args.question, args.topic)
    else:  # compare
        compare_approaches(args.question, args.topic)
    
    print(f"\n{'=' * 80}")
    print("PRUEBA COMPLETADA")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main() 