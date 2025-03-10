from typing import List, Dict, Any
import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document

# Cargar variables de entorno
load_dotenv()

# Inicializar el cliente de OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def format_documents_for_openai(documents: List[Document]) -> str:
    """
    Formatea los documentos para OpenAI.
    """
    formatted_docs = ""
    for i, doc in enumerate(documents):
        # Obtener la fuente del documento
        source = doc.metadata.get('source', f'Documento {i+1}')
        
        # Verificar si la fuente es de Pinecone
        if "pinecone_docs" in source:
            # Formatear la fuente para que sea más clara
            source = source.replace("pinecone_docs/", "Pinecone: ")
            formatted_docs += f"\n\nDOCUMENTO [{i+1}] (PINECONE): {source}\n"
        else:
            formatted_docs += f"\n\nDOCUMENTO [{i+1}]: {source}\n"
            
        formatted_docs += f"{doc.page_content}\n"
        formatted_docs += "-" * 50
    
    return formatted_docs

def generate_with_openai(question: str, documents: List[Document]) -> Dict[str, Any]:
    """
    Genera una respuesta usando OpenAI GPT-4o-mini con citas numeradas.
    """
    # Formatear documentos para OpenAI
    formatted_docs = format_documents_for_openai(documents)
    
    # Crear el sistema y el mensaje del usuario
    system_message = """Eres un asistente jurídico experto especializado en derecho tributario colombiano, con énfasis en IVA. Tu objetivo es proporcionar respuestas precisas, detalladas y fundamentadas a consultas legales, siguiendo una estructura específica y con especial atención a cambios normativos y jurisprudenciales.

ESTRUCTURA DE LA RESPUESTA:
Tu respuesta debe organizarse OBLIGATORIAMENTE en las siguientes secciones:

1. REFERENCIA:
   - Descripción breve y precisa de la consulta tributaria.
   - Identificación del tema principal y aspectos secundarios a abordar.

2. CONTENIDO:
   - Índice detallado de las secciones y subsecciones que componen tu respuesta.
   - Debe incluir todos los puntos que se desarrollarán en el análisis.

3. ENTENDIMIENTO:
   - Explicación de cómo interpretas la consulta.
   - Identificación de los aspectos clave a resolver.
   - Mención de la normativa principal aplicable.

4. CONCLUSIÓN:
   - Resumen ejecutivo de tu opinión jurídica.
   - Puntos clave de la respuesta.
   - Recomendaciones principales.

5. ANÁLISIS:
   5.1. Marco Normativo Vigente:
        - Disposiciones legales aplicables.
        - Artículos relevantes del Estatuto Tributario.
        - Normas complementarias.

   5.2. Evolución y Cambios Normativos:
        - Modificaciones relevantes en los últimos 3 años.
        - Comparación entre regulación anterior y actual.
        - Impacto práctico de los cambios.

   5.3. Jurisprudencia Relevante:
        - Sentencias clave del Consejo de Estado.
        - Cambios en interpretaciones jurisprudenciales.
        - Anulaciones de conceptos DIAN.

   5.4. Doctrina y Controversias:
        - Postura actual de la DIAN.
        - Debates interpretativos existentes.
        - Conflictos entre DIAN y Consejo de Estado.

   5.5. Consideraciones Prácticas:
        - Aplicación práctica de la normativa.
        - Riesgos y aspectos a considerar.
        - Recomendaciones detalladas.

INSTRUCCIONES SOBRE CITAS:
1. Usa el formato de cita [n] después de cada afirmación basada en los documentos.
2. Numera las citas secuencialmente: [1], [2], [3], etc.
3. Cada número debe corresponder al documento del que extraes la información.
4. Si usas información de varios documentos en una misma afirmación, incluye todas las citas relevantes: [1][2].
5. Coloca las citas inmediatamente después de la afirmación que respaldan.
6. CADA afirmación importante debe tener su correspondiente cita.

INSTRUCCIONES ESPECIALES:
1. SIEMPRE destaca los cambios normativos recientes y sus implicaciones.
2. Enfatiza cuando una interpretación de la DIAN haya sido anulada por el Consejo de Estado.
3. Señala explícitamente cuando existan controversias o diferentes interpretaciones sobre un tema.
4. Advierte sobre posibles cambios pendientes o proyectos de ley que puedan afectar la interpretación actual.
5. Incluye ejemplos prácticos cuando sea posible para ilustrar la aplicación de la norma.

Ejemplo de formato correcto:
"La tarifa general del IVA en Colombia es del 19% [1]. Sin embargo, es importante notar que el Consejo de Estado, en sentencia reciente, ha modificado la interpretación de su base gravable en ciertos casos [2], contradiciendo la postura tradicional de la DIAN [3]."

NO uses notas al pie ni referencias al final. Las citas deben estar integradas en el texto."""
    
    user_message = f"""Pregunta: {question}

DOCUMENTOS PARA CONSULTA:
{formatted_docs}

IMPORTANTE: 
1. Responde siguiendo ESTRICTAMENTE la estructura de 5 secciones principales especificada (REFERENCIA, CONTENIDO, ENTENDIMIENTO, CONCLUSIÓN, ANÁLISIS).
2. Usa el formato de citas numéricas [1], [2], etc. después de cada afirmación que hagas.
3. Cada número debe corresponder al documento del que extraes la información.
4. Asegúrate de que CADA afirmación importante tenga su correspondiente cita entre corchetes.
5. Enfatiza especialmente los cambios normativos y jurisprudenciales recientes.
6. Destaca cualquier contradicción entre la DIAN y el Consejo de Estado.
7. Mantén una numeración clara (1., 2., 3., etc. para secciones principales y 5.1., 5.2., etc. para subsecciones del análisis)."""
    
    try:
        # Llamar a la API de OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2  # Un poco de temperatura para mejorar la fluidez del texto
        )
        
        # Extraer el texto de la respuesta
        response_text = response.choices[0].message.content
        
        # Extraer citas del texto usando el patrón [1], [2], etc.
        citations = extract_citations_from_text(response_text, documents)
        
        print(f"Se extrajeron {len(citations)} citas del texto")
        
        return {
            "text": response_text,
            "citations": citations,
            "raw_message": response
        }
    
    except Exception as e:
        print(f"Error al generar respuesta con OpenAI: {str(e)}")
        # Devolver una respuesta de error
        return {
            "text": f"Lo siento, hubo un error al generar la respuesta: {str(e)}",
            "citations": [],
            "raw_message": None
        }

def extract_citations_from_text(text, documents):
    """
    Extrae citas manuales del texto en formato [1], [2], etc.
    """
    citations = []
    # Buscar patrones de cita como [1], [2], etc.
    citation_pattern = r'\[(\d+)\]'
    matches = re.finditer(citation_pattern, text)
    
    citation_indices = set()  # Para evitar duplicados
    
    for match in matches:
        citation_num = int(match.group(1))
        if citation_num <= len(documents) and citation_num not in citation_indices:
            citation_indices.add(citation_num)
            doc = documents[citation_num - 1]
            
            # Extraer un fragmento más significativo del documento
            content = doc.page_content
            excerpt = content[:200] + "..." if len(content) > 200 else content
            
            # Obtener la fuente del documento
            source = doc.metadata.get("source", f"Documento {citation_num}")
            
            # Verificar si la fuente es de Pinecone
            if "pinecone_docs" in source:
                # Formatear la fuente para que sea más clara
                source = source.replace("pinecone_docs/", "Pinecone: ")
            
            citations.append({
                "document_title": source,
                "cited_text": excerpt,
                "document_index": citation_num - 1
            })
    
    return citations 