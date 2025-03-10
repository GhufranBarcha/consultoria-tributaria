from typing import Dict, List, Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from graph.chains.answer_grader import answer_grader
from graph.chains.document_grader import document_grader
from graph.chains.generation import generation_chain
from graph.chains.retrieval import retriever, query_pinecone, chroma_retriever
from graph.chains.web_search import web_search_chain
from graph.state import GraphState
# Importar la función para generar con OpenAI
from graph.chains.openai_generation import generate_with_openai

# Importar la función debug_print
from graph.graph import debug_print

def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from the vectorstore based on the topic in the state.
    """
    debug_print("---RETRIEVE---")
    question = state["question"]
    
    # Obtener el tema seleccionado si existe en el estado
    debug_print(f"nodes.py: Estado completo: {state}")
    
    # Forzar el uso de Pinecone si estamos en la sección de Renta
    # Esto es un workaround temporal hasta que encontremos el problema real
    if "topic" in state and state["topic"] == "Renta":
        print("FORZANDO USO DE PINECONE PARA RENTA")
        print(f"Pregunta: {question}")
        documents = query_pinecone(question)
        print(f"Recuperados {len(documents)} documentos de Pinecone")
        
        # Verificar las fuentes de los documentos
        for i, doc in enumerate(documents):
            source = doc.metadata.get('source', 'Desconocido')
            print(f"Documento {i+1}: {source}")
        
        return {"documents": documents}
    else:
        # Comportamiento normal para otros temas
        topic = state.get("topic", None)
        debug_print(f"nodes.py: Tema obtenido del estado: '{topic}', tipo: {type(topic)}")
        debug_print(f"---TEMA SELECCIONADO: {topic}---")
        
        # Para otros temas o si no se especifica tema, usar Chroma
        debug_print("---USANDO CHROMA PARA OTROS TEMAS---")
        documents = chroma_retriever.invoke(question)
        debug_print(f"---RECUPERADOS {len(documents)} DOCUMENTOS DE CHROMA---")
    
        return {"documents": documents}


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Grade the documents for relevance to the question.
    """
    debug_print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    graded_documents = []
    web_search = False

    for document in documents:
        score = document_grader.invoke(
            {"document": document, "question": question}
        )
        if document_grade := score.binary_score:
            debug_print("---GRADE: DOCUMENT RELEVANT---")
            graded_documents.append(document)
        else:
            debug_print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True

    return {"documents": graded_documents, "web_search": web_search}


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response based on the question and documents.
    """
    debug_print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # Obtener el tema si existe en el estado
    topic = state.get("topic", None)
    debug_print(f"---GENERANDO RESPUESTA PARA TEMA: {topic}---")
    
    # Usar OpenAI para generar la respuesta con citas
    try:
        # Llamar a generate_with_openai para obtener una respuesta estructurada con citas
        print(f"generate: Llamando a generate_with_openai con {len(documents)} documentos")
        openai_response = generate_with_openai(question, documents)
        
        # Extraer la respuesta y las citas
        generation = openai_response["text"]
        citations = openai_response["citations"]
        
        print(f"generate: Respuesta generada con {len(citations)} citas")
        print(f"generate: Longitud de la respuesta: {len(generation)} caracteres")
        
        # Verificar si la respuesta tiene la estructura esperada
        has_structure = "REFERENCIA" in generation and "ANÁLISIS" in generation
        if has_structure:
            print("generate: La respuesta tiene la estructura esperada")
        else:
            print("generate: ADVERTENCIA - La respuesta no tiene la estructura esperada")
        
        # Imprimir las primeras 100 caracteres de la respuesta para depuración
        print(f"generate: Primeros 100 caracteres de la respuesta: {generation[:100]}...")
        
        # Devolver un diccionario con la generación y las citas
        return {
            "generation": generation,
            "citations": citations,
            "has_structure": has_structure
        }
    except Exception as e:
        debug_print(f"---ERROR WITH OPENAI: {str(e)}---")
        # Fallback a la generación original si hay error con OpenAI
        try:
            generation = generation_chain.invoke(
                {"question": question, "documents": documents}
            )
            debug_print("---FALLBACK GENERATION SUCCESSFUL---")
            return {"generation": generation}
        except Exception as inner_e:
            debug_print(f"---ERROR WITH FALLBACK GENERATION: {str(inner_e)}---")
            return {"generation": f"Error al generar respuesta: {str(e)}. Error en fallback: {str(inner_e)}"}


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Search the web for information.
    """
    debug_print("---WEB SEARCH---")
    question = state["question"]
    web_results = web_search_chain.invoke(question)
    documents = state["documents"] + web_results
    return {"documents": documents, "web_search": True} 