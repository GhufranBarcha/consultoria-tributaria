from dotenv import load_dotenv
import os
from graph.graph import app

# Cargar variables de entorno
load_dotenv()

def main():
    # Ejemplo 1: Pregunta sobre memoria de agentes (debería usar el almacén de vectores)
    question1 = "What is LLM agent memory?"
    print("\n\n===== EJEMPLO 1: PREGUNTA SOBRE MEMORIA DE AGENTES =====")
    print(f"Pregunta: {question1}")
    result1 = app.invoke(input={"question": question1})
    print(f"Respuesta: {result1['generation']}")
    print(f"¿Se usó búsqueda web?: {result1['web_search']}")
    print(f"Número de documentos encontrados: {len(result1['documents'])}")
    
    # Ejemplo 2: Pregunta sobre Claude 3.5 Sonnet (debería usar búsqueda web)
    question2 = "What are the main features of Claude 3.5 Sonnet?"
    print("\n\n===== EJEMPLO 2: PREGUNTA SOBRE CLAUDE 3.5 SONNET =====")
    print(f"Pregunta: {question2}")
    result2 = app.invoke(input={"question": question2})
    print(f"Respuesta: {result2['generation']}")
    print(f"¿Se usó búsqueda web?: {result2.get('web_search', True)}")
    print(f"Número de documentos encontrados: {len(result2['documents'])}")
    
    # Ejemplo 3: Pregunta que podría causar alucinaciones
    question3 = "Who invented LLM agents and in what year?"
    print("\n\n===== EJEMPLO 3: PREGUNTA POTENCIALMENTE ALUCINATORIA =====")
    print(f"Pregunta: {question3}")
    result3 = app.invoke(input={"question": question3})
    print(f"Respuesta: {result3['generation']}")
    print(f"¿Se usó búsqueda web?: {result3.get('web_search', True)}")
    print(f"Número de documentos encontrados: {len(result3['documents'])}")

if __name__ == "__main__":
    main() 