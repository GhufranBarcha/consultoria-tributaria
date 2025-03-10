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
        Eres un experto en derecho tributario colombiano especializado en renta. Tu tarea es sintetizar un concepto jurídico integral basado en las respuestas a varias subpreguntas.

        # Pregunta Original
        {question}

        # Respuestas a Subpreguntas
        {subquestion_answers}

        # Instrucciones
        Genera un concepto jurídico estructurado que responda la pregunta original de manera exhaustiva. 
        Tu respuesta DEBE seguir EXACTAMENTE esta estructura:

        1. REFERENCIA
        - Título descriptivo del concepto (relacionado con la consulta)
        - Fecha de elaboración
        - Tema principal y subtemas

        2. CONTENIDO
        - Tabla de contenido detallada
        - Debe listar TODAS las secciones y subsecciones que se desarrollarán
        - Usar numeración exacta (1., 2., 3. para secciones principales; 5.1., 5.2., etc. para subsecciones)

        3. ENTENDIMIENTO
        - DEBE comenzar con la frase "El cliente desea conocer"
        - Explicar claramente qué se entiende de la consulta
        - Si hay varios aspectos a resolver, enumerarlos (3.1., 3.2., etc.)
        - Identificar el marco normativo aplicable

        4. CONCLUSIÓN
        - Respuestas concretas y directas a cada aspecto consultado
        - OBLIGATORIAMENTE usar enumeración (4.1., 4.2., etc.)
        - Cada conclusión debe ser autónoma y comprensible por sí misma
        - Incluir recomendaciones prácticas cuando aplique

        5. ANÁLISIS
        Desarrollar OBLIGATORIAMENTE estas subsecciones:
        5.1. Marco Normativo Vigente
            - Normas aplicables
            - Artículos específicos
            - Jerarquía normativa
        5.2. Interpretación de la DIAN
            - Doctrina vigente
            - Conceptos relevantes
            - Cambios de interpretación
        5.3. Jurisprudencia Relevante
            - Sentencias del Consejo de Estado
            - Precedentes vinculantes
            - Cambios jurisprudenciales
        5.4. Análisis Práctico
            - Aplicación al caso concreto
            - Ejemplos cuando sea posible
            - Consideraciones especiales
        5.5. Riesgos y Recomendaciones
            - Identificación de riesgos
            - Estrategias de mitigación
            - Recomendaciones específicas

        REGLAS OBLIGATORIAS:
        1. DEBES mantener EXACTAMENTE la estructura y numeración indicada
        2. DEBES incluir todas las secciones y subsecciones en el orden especificado
        3. Cada afirmación importante debe tener su respectiva cita [X]
        4. Las citas deben aparecer inmediatamente después de cada afirmación
        5. El formato de citas es [1], [2], etc., usando corchetes
        6. NO uses notas al pie ni referencias al final
        7. Mantén un estilo formal y técnico
        8. Usa viñetas (-) para listar elementos dentro de cada sección
        9. Si no hay información suficiente para alguna sección, indícalo explícitamente
        10. La respuesta debe ser autosuficiente (no requerir información adicional)

        IMPORTANTE: Si no sigues EXACTAMENTE esta estructura o formato, la respuesta será rechazada.
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
    
    1. REFERENCIA
    - Título del concepto: [título descriptivo]
    - Fecha de elaboración: [fecha actual]
    - Tema principal: [tema específico]
    - Subtemas: [lista con viñetas]

    2. CONTENIDO
    Debe ser una tabla de contenido clara y completa con este formato exacto:
    1. REFERENCIA
    2. CONTENIDO
    3. ENTENDIMIENTO
       3.1. [Primer aspecto]
       3.2. [Segundo aspecto]
       ...
    4. CONCLUSIÓN
       4.1. [Primera conclusión]
       4.2. [Segunda conclusión]
       ...
    5. ANÁLISIS
       5.1. Marco Normativo Vigente
       5.2. Interpretación de la DIAN
       5.3. Jurisprudencia Relevante
       5.4. Análisis Práctico
       5.5. Riesgos y Recomendaciones

    3. ENTENDIMIENTO
    - DEBE iniciar con "El cliente desea conocer"
    - Usar enumeración 3.1., 3.2., etc. para cada aspecto
    - Identificar claramente el marco normativo aplicable
    - NO incluir citas en esta sección

    4. CONCLUSIÓN
    - Usar SIEMPRE enumeración (4.1., 4.2., etc.)
    - Cada conclusión debe ser una respuesta directa
    - Máximo 3-4 líneas por conclusión
    - Incluir citas clave [X] que respalden cada conclusión

    5. ANÁLISIS
    Desarrollar OBLIGATORIAMENTE en este orden:
    5.1. Marco Normativo Vigente
        - Normas específicas con artículos
        - Jerarquía normativa clara
        - Citar la fuente de cada norma [X]
    
    5.2. Interpretación de la DIAN
        - Doctrina actual vigente
        - Conceptos específicos con número y fecha
        - Cambios de interpretación si existen
    
    5.3. Jurisprudencia Relevante
        - Sentencias específicas con referencias
        - Precedentes vinculantes
        - Cambios jurisprudenciales
    
    5.4. Análisis Práctico
        - Ejemplos concretos
        - Casos prácticos
        - Aplicación específica
    
    5.5. Riesgos y Recomendaciones
        - Riesgos identificados
        - Estrategias de mitigación
        - Recomendaciones específicas

    REGLAS DE FORMATO:
    1. Espaciado:
       - Una línea en blanco entre secciones principales
       - No dejar líneas en blanco dentro de una misma sección
       - Usar viñetas (-) para listas dentro de secciones
    
    2. Numeración:
       - Secciones principales: 1., 2., 3., 4., 5.
       - Subsecciones: 3.1., 3.2., 5.1., 5.2., etc.
       - NO usar otros formatos de numeración
    
    3. Citas:
       - Formato único: [1], [2], etc.
       - Una cita por referencia (NO usar [1,2] o [1][2])
       - Cita inmediatamente después de la afirmación
       - Mínimo una cita por párrafo en secciones 4 y 5
    
    4. Contenido:
       - Párrafos cortos (máximo 4 líneas)
       - Lenguaje técnico pero claro
       - Sin abreviaturas no explicadas
       - Sin referencias cruzadas

    VERIFICA ANTES DE FINALIZAR:
    1. ¿Están todas las secciones principales (1-5)?
    2. ¿Cada sección tiene sus subsecciones completas?
    3. ¿La numeración es correcta y consistente?
    4. ¿Hay suficientes citas y están bien formateadas?
    5. ¿El formato visual es claro y consistente?
    6. ¿Las conclusiones son directas y están enumeradas?
    7. ¿El entendimiento comienza con "El cliente desea conocer"?
    8. ¿La tabla de contenido refleja exactamente las secciones?

    IMPORTANTE: Si no cumples TODAS estas reglas, la respuesta será rechazada.
    
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
    Procesa una consulta utilizando un enfoque simplificado para generar un concepto jurídico estructurado.
    
    Args:
        question: La consulta del usuario
        topic: El tema de la consulta (IVA o Renta)
        
    Returns:
        Un diccionario con la respuesta y metadatos
    """
    import re  # Añadir importación de re para expresiones regulares
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    print(f"Procesando consulta con flujo simplificado: {question}")
    
    try:
        # Paso 1: Recuperar documentos relevantes directamente
        print("Recuperando documentos relevantes...")
        if topic and topic.lower() == "renta":
            documents = query_pinecone(question)
        else:
            documents = retriever.invoke(query=question, topic=topic)
        
        if not documents:
            print("No se encontraron documentos relevantes.")
            return {
                "question": question,
                "topic": topic,
                "final_answer": "Lo siento, no se encontraron documentos relevantes para responder a tu consulta. Por favor, intenta reformular la pregunta o consulta sobre otro tema.",
                "reviewed_answer": "Lo siento, no se encontraron documentos relevantes para responder a tu consulta. Por favor, intenta reformular la pregunta o consulta sobre otro tema.",
                "subquestions": [],
                "citations": [],
                "all_documents": [],
                "debug_info": {"error": "no_documents_found"}
            }
        
        # Formatear documentos para el modelo
        formatted_docs = ""
        all_docs = []
        for i, doc in enumerate(documents):
            # Obtener la fuente del documento
            source = doc.metadata.get('source', f'Documento {i+1}')
            title = source.split('/')[-1] if '/' in source else source
            
            # Guardar documento para referencias
            all_docs.append({
                "id": i + 1,
                "title": title,
                "content": doc.page_content,
                "source": source
            })
            
            # Formatear para el prompt
            formatted_docs += f"\n\nDOCUMENTO [{i+1}]: {source}\n"
            formatted_docs += f"{doc.page_content}\n"
            formatted_docs += "-" * 50
        
        # Paso 2: Generar respuesta estructurada en un solo paso
        print("Generando concepto jurídico estructurado...")
        
        # Usar un modelo más potente para la generación
        concept_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        
        # Prompt simplificado pero completo
        concept_prompt = ChatPromptTemplate.from_template("""
        Eres un experto en derecho tributario colombiano especializado en {topic}.
        
        Debes elaborar un concepto jurídico estructurado que responda a la siguiente consulta:
        
        CONSULTA: {question}
        
        DOCUMENTOS DE REFERENCIA:
        {documents}
        
        ESTRUCTURA OBLIGATORIA DEL CONCEPTO:
        
        1. REFERENCIA
        - Título del concepto: [título descriptivo relacionado con la consulta]
        - Fecha de elaboración: [fecha actual]
        - Tema principal: [tema específico]
        - Subtemas: [lista con viñetas]

        2. CONTENIDO
        [Tabla de contenido exacta con todas las secciones y subsecciones]

        3. ENTENDIMIENTO
        - Debe comenzar con "El cliente desea conocer"
        - Explicación clara de la consulta
        - Aspectos clave a resolver (usar 3.1, 3.2, etc.)

        4. CONCLUSIÓN
        - Respuestas directas y enumeradas (4.1, 4.2, etc.)
        - Cada conclusión debe ser concreta y respaldada con citas

        5. ANÁLISIS
        5.1. Marco Normativo Vigente
            - Normas aplicables con artículos específicos
            - Jerarquía normativa
        5.2. Interpretación de la DIAN
            - Doctrina vigente y conceptos relevantes
            - Cambios de interpretación si existen
        5.3. Jurisprudencia Relevante
            - Sentencias específicas del Consejo de Estado
            - Precedentes vinculantes
        5.4. Análisis Práctico
            - Aplicación al caso concreto
            - Ejemplos ilustrativos
        5.5. Riesgos y Recomendaciones
            - Riesgos identificados
            - Estrategias y recomendaciones

        REGLAS DE FORMATO:
        1. Usa numeración clara y consistente (1., 2., 3. para secciones principales; 3.1., 3.2., 5.1., 5.2., etc. para subsecciones)
        2. Incluye citas con formato [1], [2], etc. después de cada afirmación importante
        3. Usa viñetas (-) para listas dentro de secciones
        4. Mantén un estilo formal y técnico
        5. Párrafos cortos y concisos
        
        IMPORTANTE: Asegúrate de incluir TODAS las secciones en el orden indicado y con la numeración correcta.
        """)
        
        # Generar el concepto jurídico
        chain = concept_prompt | concept_llm | StrOutputParser()
        concept = chain.invoke({
            "question": question,
            "documents": formatted_docs,
            "topic": topic or "derecho tributario"
        })
        
        # Paso 3: Extraer citas
        print("Extrayendo citas...")
        citation_pattern = r'\[(\d+)\]'
        citation_numbers = set(re.findall(citation_pattern, concept))
        
        citations = []
        for num in citation_numbers:
            num_idx = int(num) - 1
            if num_idx < len(all_docs):
                doc = all_docs[num_idx]
                citations.append({
                    "document_title": doc["title"],
                    "cited_text": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"]
                })
        
        # Devolver resultado
        return {
            "question": question,
            "topic": topic,
            "final_answer": concept,  # No hay necesidad de tener respuesta revisada separada
            "reviewed_answer": concept,  # Mantener para compatibilidad
            "subquestions": [],  # Ya no generamos subpreguntas
            "citations": citations,
            "all_documents": all_docs,
            "debug_info": {"approach": "simplified_direct_generation"}
        }
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error en process_with_experimental_workflow: {str(e)}")
        print(error_traceback)
        
        # Devolver un mensaje de error amigable
        return {
            "question": question,
            "topic": topic,
            "final_answer": f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}",
            "reviewed_answer": f"Lo siento, ocurrió un error al procesar tu consulta: {str(e)}",
            "subquestions": [],
            "citations": [],
            "all_documents": [],
            "debug_info": {"error": str(e), "traceback": error_traceback}
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de consulta
    sample_question = "¿Cuáles son los requisitos para solicitar la devolución de IVA en Colombia?"
    
    # Procesar con el flujo simplificado
    result = process_with_experimental_workflow(sample_question, "IVA")
    
    # Imprimir resultado
    print("\n" + "=" * 50)
    print("CONSULTA ORIGINAL:")
    print(sample_question)
    print("\n" + "=" * 50)
    print("RESPUESTA FINAL:")
    print(result["final_answer"])
    print("\n" + "=" * 50) 