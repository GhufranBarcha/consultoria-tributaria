"""
Flujo de trabajo experimental que implementa un enfoque de agentes para procesar consultas jurídicas.

Este módulo implementa un flujo alternativo que:
1. Descompone la consulta principal en subpreguntas
2. Responde cada subpregunta individualmente
3. Sintetiza las respuestas en una respuesta final
4. Revisa y refina la respuesta final

Este flujo es independiente de la aplicación principal y puede ser probado sin afectar
la funcionalidad existente.
"""

import os
from typing import Dict, List, Any, Optional, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import json

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Importar componentes existentes para reutilizarlos
from graph.chains.retrieval import retriever, query_pinecone
from graph.chains.openai_generation import generate_with_openai

# Cargar variables de entorno
load_dotenv()

# Definir el estado como un TypedDict para compatibilidad con LangGraph
class ExperimentalState(TypedDict, total=False):
    """Estado del grafo experimental."""
    question: str
    topic: Optional[str]
    subquestions: List[Dict]
    all_documents: List[Dict]
    final_answer: Optional[str]
    reviewed_answer: Optional[str]
    citations: List[Dict]
    debug_info: Dict

# Configurar modelos de lenguaje
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
reviewer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Nodo 1: Descomponer la consulta en subpreguntas
def decompose_question(state: ExperimentalState) -> ExperimentalState:
    """Descompone la consulta principal en subpreguntas más específicas."""
    print(f"Descomponiendo la consulta: {state['question']}")
    
    # Prompt para descomponer la consulta
    decompose_prompt = ChatPromptTemplate.from_template("""
    Eres un experto en derecho tributario colombiano especializado en {topic}.
    
    Necesito que analices la siguiente consulta jurídica y la descompongas en subpreguntas más específicas 
    que ayuden a responderla de manera completa y precisa.
    
    Consulta: {question}
    
    Genera entre 2 y 4 subpreguntas que:
    1. Aborden diferentes aspectos de la consulta principal
    2. Sean específicas y concretas
    3. En conjunto, permitan responder completamente la consulta original
    
    Devuelve ÚNICAMENTE un array JSON con las subpreguntas, con este formato exacto:
    [
        {{"id": "subq1", "text": "Primera subpregunta"}},
        {{"id": "subq2", "text": "Segunda subpregunta"}},
        ...
    ]
    """)
    
    # Parsear la salida como JSON
    parser = JsonOutputParser()
    
    # Crear la cadena de procesamiento
    chain = decompose_prompt | planner_llm | parser
    
    # Ejecutar la cadena
    subquestions_data = chain.invoke({
        "question": state["question"],
        "topic": state.get("topic") or "derecho tributario"
    })
    
    # Convertir los datos a diccionarios
    subquestions = []
    for sq in subquestions_data:
        subquestions.append({
            "id": sq["id"],
            "text": sq["text"],
            "answered": False,
            "answer": None,
            "documents": []
        })
    
    # Actualizar el estado
    new_state = state.copy()
    new_state["subquestions"] = subquestions
    if "debug_info" not in new_state:
        new_state["debug_info"] = {}
    new_state["debug_info"]["decompose"] = f"Generadas {len(subquestions)} subpreguntas"
    
    return new_state

# Nodo 2: Recuperar documentos para cada subpregunta
def retrieve_for_subquestions(state: ExperimentalState) -> ExperimentalState:
    """Recupera documentos relevantes para cada subpregunta."""
    print("Recuperando documentos para subpreguntas")
    
    # Importar el MultiRetriever
    from graph.chains.retrieval import retriever
    
    # Lista para almacenar todos los documentos recuperados
    all_documents = []
    
    # Para cada subpregunta, recuperar documentos
    for i, subq in enumerate(state["subquestions"]):
        print(f"Recuperando para subpregunta {i+1}: {subq['text']}")
        
        try:
            # Usar el MultiRetriever que selecciona automáticamente entre Chroma y Pinecone
            docs = retriever.invoke(subq["text"], state.get("topic"))
            
            # Procesar los documentos para asegurar que tengan la información necesaria
            processed_docs = []
            for j, doc in enumerate(docs):
                # Extraer el contenido y metadatos
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                
                # Crear un documento procesado con la información necesaria
                processed_doc = {
                    "id": f"doc_{i}_{j}",
                    "title": metadata.get('source', f"Documento {len(all_documents) + 1}"),
                    "content": content,
                    "source": metadata.get('source', "Desconocido"),
                    "metadata": metadata
                }
                
                processed_docs.append(processed_doc)
                all_documents.append(processed_doc)
            
            # Actualizar la subpregunta con los documentos recuperados
            subq["documents"] = processed_docs
        except Exception as e:
            print(f"Error al recuperar documentos: {str(e)}")
    
    # Actualizar el estado
    new_state = state.copy()
    new_state["subquestions"] = state["subquestions"]  # Ya actualizado con los documentos
    new_state["all_documents"] = all_documents
    new_state["debug_info"]["retrieve"] = f"Recuperados {len(all_documents)} documentos en total"
    
    return new_state

# Nodo 3: Responder cada subpregunta
def answer_subquestions(state: ExperimentalState) -> ExperimentalState:
    """Responde cada subpregunta utilizando los documentos recuperados."""
    print("Respondiendo subpreguntas")
    
    answer_prompt = ChatPromptTemplate.from_template("""
    Eres un experto en derecho tributario colombiano especializado en {topic}.
    
    Responde la siguiente subpregunta utilizando ÚNICAMENTE la información proporcionada en los documentos.
    
    Subpregunta: {subquestion}
    
    Documentos:
    {documents}
    
    Instrucciones:
    1. Responde de manera clara, precisa y directa
    2. Utiliza SOLO la información de los documentos proporcionados
    3. Si los documentos no contienen información suficiente, indica claramente qué información falta
    4. Incluye referencias a los documentos utilizando el formato [Documento X]
    5. No inventes información ni cites fuentes que no estén en los documentos
    
    Tu respuesta:
    """)
    
    new_state = state.copy()
    
    for i, subq in enumerate(state["subquestions"]):
        print(f"Respondiendo subpregunta {i+1}: {subq['id']}")
        
        # Preparar los documentos para esta subpregunta
        docs_text = ""
        for j, doc in enumerate(subq["documents"][:3]):  # Limitamos a 3 documentos por subpregunta
            docs_text += f"\nDocumento {j+1}:\n"
            docs_text += f"Fuente: {doc.get('source', 'Desconocida')}\n"
            docs_text += f"{doc.get('content', '')}\n"
            docs_text += "-" * 50 + "\n"
        
        # Si no hay documentos, usar un mensaje específico
        if not docs_text:
            docs_text = "No se encontraron documentos relevantes para esta subpregunta."
        
        # Generar respuesta
        chain = answer_prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "subquestion": subq["text"],
            "documents": docs_text,
            "topic": state.get("topic") or "derecho tributario"
        })
        
        # Actualizar el estado
        new_state["subquestions"][i]["answer"] = answer
        new_state["subquestions"][i]["answered"] = True
    
    new_state["debug_info"]["answer_subquestions"] = f"Respondidas {len(new_state['subquestions'])} subpreguntas"
    return new_state

# Nodo 4: Sintetizar respuestas en una respuesta final
def synthesize_answer(state: ExperimentalState) -> ExperimentalState:
    """
    Sintetiza las respuestas a las subpreguntas en una respuesta final.
    """
    print("Sintetizando respuesta final")
    
    # Preparar las respuestas a las subpreguntas
    subquestion_answers = []
    for i, subq in enumerate(state["subquestions"]):
        subquestion_answers.append(f"Subpregunta {i+1}: {subq['text']}\nRespuesta: {subq['answer']}")
    
    all_subquestion_answers = "\n\n".join(subquestion_answers)
    
    # Preparar el prompt para la respuesta final
    prompt = PromptTemplate.from_template(
        """
        You are a legal expert specialized in Colombian tax law. Your task is to synthesize a comprehensive legal concept based on the answers to several sub-questions.

        # Original Question
        {question}

        # Answers to Sub-questions
        {subquestion_answers}

        # Instructions
        Create a structured legal concept that addresses the original question comprehensively. Your response should follow this structure:

        1. REFERENCE: Brief description of the query or consultation.
        
        2. CONTENT: A table of contents listing all sections and subsections of your response.
        
        3. UNDERSTANDING: Explanation of how you understand the query, identifying key aspects to address.
        
        4. CONCLUSION: Summary of your legal opinion on the matter.
        
        5. ANALYSIS: Detailed examination of the legal issues, including:
           - Relevant legal provisions
           - Interpretation of those provisions
           - Application to the specific case
           - Considerations of doctrine and jurisprudence when relevant

        Format your response with clear enumeration (1., 2., 3., etc. for main sections and 3.1., 3.2., etc. for subsections).
        
        Ensure your response is well-structured, legally sound, and directly addresses the original question.
        """
    )
    
    # Generar la respuesta final
    chain = prompt | llm
    response = chain.invoke({
        "question": state["question"],
        "subquestion_answers": all_subquestion_answers
    })
    
    # Actualizar el estado
    new_state = state.copy()
    new_state["final_answer"] = response.content
    new_state["debug_info"]["synthesize"] = "Respuesta final sintetizada"
    
    return new_state

# Nodo 5: Revisar y refinar la respuesta
def review_answer(state: ExperimentalState) -> ExperimentalState:
    """Revisa y refina la respuesta final para mejorar su calidad."""
    print("Revisando respuesta final")
    
    review_prompt = ChatPromptTemplate.from_template("""
    Eres un revisor experto en derecho tributario colombiano especializado en {topic}.
    
    Revisa el siguiente concepto jurídico y mejóralo en términos de:
    1. Precisión técnica y jurídica
    2. Claridad y estructura
    3. Fundamentación en la normativa
    4. Completitud (que responda todos los aspectos de la consulta)
    
    Consulta original: {question}
    
    Concepto jurídico a revisar:
    {answer}
    
    ESTRUCTURA REQUERIDA DEL CONCEPTO JURÍDICO:
    
    1. REFERENCIA: Título del concepto (relacionado con la consulta).
    
    2. CONTENIDO: Tabla de contenido que liste todas las secciones del concepto.
    
    3. ENTENDIMIENTO: Debe comenzar con "El cliente desea conocer" y explicar lo que se entiende de la consulta.
       Si hay varios aspectos, enuméralos como 3.1, 3.2, etc.
    
    4. CONCLUSIÓN: Conclusiones concretas y enumeradas (4.1, 4.2, etc.).
    
    5. ANÁLISIS: Desarrollo detallado del análisis jurídico, citando las fuentes.
       Debe estar subdividido y enumerado (5.1, 5.2, etc.).
    
    INSTRUCCIONES PARA LA REVISIÓN:
    1. Asegúrate de que se mantenga EXACTAMENTE la estructura numerada requerida
    2. Verifica que todas las afirmaciones estén fundamentadas con citas a documentos [X]
    3. Mejora la redacción y precisión técnica sin alterar la estructura
    4. Asegúrate de que el entendimiento comience con "El cliente desea conocer"
    5. Verifica que las conclusiones sean concretas y estén enumeradas
    6. Confirma que el análisis esté subdividido y enumerado correctamente
    7. IMPORTANTE: Incluye al menos 5-8 citas a documentos usando el formato [X] donde X es un número
    
    Concepto jurídico revisado y mejorado:
    """)
    
    # Generar respuesta revisada
    chain = review_prompt | reviewer_llm | StrOutputParser()
    reviewed_answer = chain.invoke({
        "question": state["question"],
        "answer": state["final_answer"],
        "topic": state.get("topic") or "derecho tributario"
    })
    
    # Actualizar el estado
    new_state = state.copy()
    new_state["reviewed_answer"] = reviewed_answer
    new_state["debug_info"]["review"] = "Respuesta final revisada y refinada"
    
    # Extraer citas (implementación mejorada)
    citation_pattern = r'\[(\d+)\]'
    import re
    citation_numbers = re.findall(citation_pattern, reviewed_answer)
    
    # Crear citas basadas en los documentos recuperados
    citations = []
    all_docs = state.get("all_documents", [])
    
    # Si tenemos documentos recuperados, usarlos para las citas
    if all_docs:
        for num in set(citation_numbers):
            num_idx = int(num) - 1
            if num_idx < len(all_docs):
                doc = all_docs[num_idx]
                citations.append({
                    "document_title": doc.get("title", f"Documento {num}"),
                    "cited_text": doc.get("content", f"Texto citado del documento {num}")[:200] + "..."
                })
            else:
                # Si no hay suficientes documentos, crear una cita ficticia
                citations.append({
                    "document_title": f"Documento {num}",
                    "cited_text": f"Texto citado del documento {num}"
                })
    else:
        # Si no hay documentos recuperados, crear citas ficticias
        for num in set(citation_numbers):
            citations.append({
                "document_title": f"Documento {num}",
                "cited_text": f"Texto citado del documento {num}"
            })
    
    new_state["citations"] = citations
    
    return new_state

# Definir el grafo experimental
def create_experimental_workflow():
    """Crea y devuelve el grafo de flujo experimental."""
    # Definir el grafo
    workflow = StateGraph(ExperimentalState)
    
    # Agregar nodos
    workflow.add_node("decompose", decompose_question)
    workflow.add_node("retrieve", retrieve_for_subquestions)
    workflow.add_node("answer_subquestions", answer_subquestions)
    workflow.add_node("synthesize", synthesize_answer)
    workflow.add_node("review", review_answer)
    
    # Definir el flujo
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "answer_subquestions")
    workflow.add_edge("answer_subquestions", "synthesize")
    workflow.add_edge("synthesize", "review")
    workflow.add_edge("review", END)
    
    # Compilar el grafo
    return workflow.compile()

# Crear el grafo experimental
experimental_app = create_experimental_workflow()

# Función para invocar el flujo experimental
def process_with_experimental_workflow(question: str, topic: str = None) -> Dict:
    """
    Procesa una consulta utilizando el flujo de trabajo experimental.
    
    Args:
        question: La consulta del usuario
        topic: El tema de la consulta (IVA o Renta)
        
    Returns:
        Un diccionario con la respuesta y metadatos
    """
    print(f"Procesando consulta con flujo experimental: {question}")
    
    # Crear estado inicial como un diccionario simple
    initial_state: ExperimentalState = {
        "question": question,
        "topic": topic,
        "subquestions": [],
        "all_documents": [],
        "final_answer": None,
        "reviewed_answer": None,
        "citations": [],
        "debug_info": {}
    }
    
    # Invocar el grafo
    result = experimental_app.invoke(initial_state)
    
    # Formatear resultado
    return {
        "question": question,
        "topic": topic,
        "final_answer": result["final_answer"],
        "reviewed_answer": result["reviewed_answer"],
        "subquestions": result["subquestions"],
        "citations": result["citations"],
        "all_documents": result["all_documents"],
        "debug_info": result["debug_info"]
    }

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de consulta
    sample_question = "¿Cuáles son los requisitos para solicitar la devolución de IVA en Colombia?"
    
    # Procesar con el flujo experimental
    result = process_with_experimental_workflow(sample_question, "IVA")
    
    # Imprimir resultado
    print("\n" + "=" * 50)
    print("CONSULTA ORIGINAL:")
    print(sample_question)
    print("\n" + "=" * 50)
    print("SUBPREGUNTAS GENERADAS:")
    for sq in result["subquestions"]:
        print(f"- {sq['text']}")
    print("\n" + "=" * 50)
    print("RESPUESTA FINAL:")
    print(result["final_answer"])
    print("\n" + "=" * 50) 