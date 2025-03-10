#!/usr/bin/env python
"""
Script para probar el flujo de trabajo experimental sin modificar la aplicación principal.
Este script permite comparar los resultados del flujo original con el experimental.
"""

import time
import argparse
from dotenv import load_dotenv
import traceback

# Importar el flujo experimental
from experimental_workflow import process_with_experimental_workflow

# Importar el flujo original para comparación
from graph.graph import app as original_app

# Cargar variables de entorno
load_dotenv()

def test_original_workflow(question: str, topic: str = None):
    """Prueba el flujo de trabajo original."""
    print(f"\n{'=' * 80}")
    print(f"PROBANDO FLUJO ORIGINAL CON: '{question}'")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    
    # Invocar el grafo original
    result = original_app.invoke(input={"question": question, "topic": topic})
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nRespuesta generada en {duration:.2f} segundos")
    print(f"\n{'=' * 40}")
    print("RESPUESTA DEL FLUJO ORIGINAL:")
    print(f"{'=' * 40}")
    print(result.get("generation", "No se generó respuesta"))
    
    # Mostrar documentos recuperados
    if "documents" in result:
        print(f"\n{'=' * 40}")
        print(f"DOCUMENTOS RECUPERADOS ({len(result['documents'])})")
        print(f"{'=' * 40}")
        for i, doc in enumerate(result["documents"][:3]):  # Mostrar solo los primeros 3
            print(f"\nDocumento {i+1}: {doc.metadata.get('source', 'Desconocido')}")
            print(f"{doc.page_content[:300]}...")  # Mostrar solo los primeros 300 caracteres
    
    return result

def test_experimental_workflow(question: str, topic: str = None):
    """Prueba el flujo experimental."""
    print("\n" + "="*80)
    print(f"EJECUTANDO FLUJO EXPERIMENTAL PARA: '{question}'")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Procesar con el flujo experimental
        result = process_with_experimental_workflow(question, topic)
        
        # Mostrar las subpreguntas generadas
        print("\n" + "-"*40)
        print("SUBPREGUNTAS GENERADAS:")
        print("-"*40)
        for i, subq in enumerate(result["subquestions"]):
            print(f"{i+1}. {subq['text']}")
        
        # Mostrar la respuesta final
        print("\n" + "-"*40)
        print("CONCEPTO JURÍDICO GENERADO:")
        print("-"*40)
        
        if "reviewed_answer" in result and result["reviewed_answer"]:
            print(result["reviewed_answer"])
        else:
            print("No se generó una respuesta revisada.")
            if "final_answer" in result and result["final_answer"]:
                print("\nRespuesta sin revisar:")
                print(result["final_answer"])
        
        # Mostrar las citas
        if "citations" in result and result["citations"]:
            print("\n" + "-"*40)
            print("REFERENCIAS:")
            print("-"*40)
            for i, citation in enumerate(result["citations"]):
                print(f"[{i+1}] {citation['document_title']}")
                print(f'    "{citation["cited_text"]}"')
        
        # Mostrar tiempo de ejecución
        end_time = time.time()
        print("\n" + "-"*40)
        print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")
        print("-"*40)
        
        return result
    except Exception as e:
        print(f"Error en el flujo experimental: {str(e)}")
        traceback.print_exc()
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Prueba el flujo de trabajo experimental")
    parser.add_argument("--question", "-q", type=str, 
                        default="¿Cuáles son los requisitos para solicitar la devolución de IVA en Colombia?",
                        help="Consulta a procesar")
    parser.add_argument("--topic", "-t", type=str, default="IVA", 
                        choices=["IVA", "Renta"],
                        help="Tema de la consulta (IVA o Renta)")
    parser.add_argument("--mode", "-m", type=str, default="both",
                        choices=["original", "experimental", "both"],
                        help="Modo de prueba (original, experimental o ambos)")
    
    args = parser.parse_args()
    
    if args.mode in ["original", "both"]:
        test_original_workflow(args.question, args.topic)
    
    if args.mode in ["experimental", "both"]:
        test_experimental_workflow(args.question, args.topic)
    
    print(f"\n{'=' * 80}")
    print("PRUEBA COMPLETADA")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main() 