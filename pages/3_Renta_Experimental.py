import streamlit as st
from dotenv import load_dotenv
import os
import time
import re
from experimental_workflow import process_with_experimental_workflow

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Consultas sobre Renta (Concepto Estructurado)",
    page_icon="",
    layout="wide"
)

# Título de la página
st.title("Consultas sobre Renta (Concepto Estructurado)")

# Descripción de la página
st.markdown("""
Esta sección le permite realizar consultas específicas sobre el Impuesto de Renta en Colombia.
La respuesta se generará como un concepto jurídico estructurado con Referencia, Contenido, Entendimiento, Conclusión y Análisis.
""")

# Verificar si la colección existe
try:
    import pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "ejhr")
    
    if not pinecone_api_key:
        st.warning("No se ha configurado la API key de Pinecone. Por favor, configura la variable PINECONE_API_KEY en el archivo .env.")
    else:
        # Inicializar Pinecone
        pc = pinecone.Pinecone(api_key=pinecone_api_key)
        existing_indexes = [index.name for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            st.warning(f"El índice {index_name} no existe en Pinecone. Por favor, crea el índice primero.")
        else:
            # Inicializar estado de sesión para Renta Experimental
            if "renta_exp_messages" not in st.session_state:
                st.session_state.renta_exp_messages = []
            
            # Función para formatear el texto con citas numeradas
            def formatear_texto_con_citas(texto, citas):
                """
                Formatea el texto con citas numeradas en HTML.
                """
                if not citas:
                    return texto
                
                # Patrón para encontrar citas en formato [X]
                patron_citas = r'\[(\d+)\]'
                
                # Reemplazar cada cita con un enlace
                def reemplazar_cita(match):
                    num_cita = match.group(1)
                    return f'<sup><a href="#cita-{num_cita}" title="Ver cita {num_cita}">[{num_cita}]</a></sup>'
                
                texto_formateado = re.sub(patron_citas, reemplazar_cita, texto)
                
                # Convertir títulos a sentence case (primera letra mayúscula, resto minúsculas)
                def to_sentence_case(match):
                    text = match.group(1)
                    # Separar el número del título
                    parts = re.match(r'(\d+\.?\d*\.?\s+)(.+)', text)
                    if parts:
                        num = parts.group(1)
                        title = parts.group(2)
                        # Convertir a sentence case (primera letra mayúscula, resto minúsculas)
                        title = title[0].upper() + title[1:].lower() if title else ""
                        return num + title
                    return text
                
                # Formatear secciones numeradas con estilos
                # Títulos principales (1., 2., 3., etc.)
                patron_titulo_principal = r'^(\d+\.\s+[A-ZÑÁÉÍÓÚÜ][^:]*):?$'
                # Primero convertir a sentence case
                texto_formateado = re.sub(patron_titulo_principal, to_sentence_case, texto_formateado, flags=re.MULTILINE)
                # Luego aplicar el formato HTML
                patron_titulo_principal = r'^(\d+\.\s+[A-ZÑÁÉÍÓÚÜa-zñáéíóúü][^:]*):?$'
                texto_formateado = re.sub(patron_titulo_principal, r'<div style="color:#2c3e50;font-weight:bold;font-size:1rem;">\1</div>', texto_formateado, flags=re.MULTILINE)
                
                # Subtítulos (X.Y., como 3.1., 4.2., etc.)
                patron_subtitulo = r'^(\d+\.\d+\.?\s+[^:]*):?$'
                # Primero convertir a sentence case
                texto_formateado = re.sub(patron_subtitulo, to_sentence_case, texto_formateado, flags=re.MULTILINE)
                # Luego aplicar el formato HTML
                patron_subtitulo = r'^(\d+\.\d+\.?\s+[A-ZÑÁÉÍÓÚÜa-zñáéíóúü][^:]*):?$'
                texto_formateado = re.sub(patron_subtitulo, r'<div style="color:#34495e;font-weight:bold;margin-left:20px;font-size:1rem;">\1</div>', texto_formateado, flags=re.MULTILINE)
                
                # Formatear la tabla de contenido como una lista sin negrilla
                patron_contenido = r'<div style="color:#2c3e50;font-weight:bold;font-size:1rem;">2\. Contenido:</div>(.*?)(?=<div style="color:#2c3e50;font-weight:bold;font-size:1rem;">3\.)'
                
                def formatear_contenido(match):
                    contenido_texto = match.group(1).strip()
                    # Convertir el contenido en una lista HTML
                    contenido_formateado = '<div style="margin-left:20px;"><ul style="list-style-type:disc;padding-left:20px;font-weight:normal;">'
                    
                    # Dividir por elementos de la tabla de contenido y formatear cada uno
                    elementos = re.findall(r'\d+\.(?:\d+\.)?\s+[^\d]+(?=\d+\.|\Z)', contenido_texto)
                    if not elementos and contenido_texto:
                        elementos = [contenido_texto]
                    
                    for elemento in elementos:
                        if elemento.strip():
                            contenido_formateado += f'<li style="margin-bottom:5px;">{elemento.strip()}</li>'
                    
                    contenido_formateado += '</ul></div>'
                    return f'<div style="color:#2c3e50;font-weight:bold;font-size:1rem;">2. Contenido:</div>{contenido_formateado}'
                
                texto_formateado = re.sub(patron_contenido, formatear_contenido, texto_formateado, flags=re.DOTALL)
                
                return texto_formateado
            
            # Mostrar mensajes anteriores
            for message in st.session_state.renta_exp_messages:
                with st.chat_message(message["role"]):
                    # Si hay citas, formatear el texto con ellas
                    if message["role"] == "assistant" and "citations" in message and message["citations"]:
                        formatted_content = formatear_texto_con_citas(message["content"], message["citations"])
                        st.markdown(formatted_content, unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
                    
                    # Si hay documentos, mostrarlos
                    if "documents" in message:
                        with st.expander("Ver fuentes utilizadas"):
                            for i, doc in enumerate(message["documents"]):
                                source = doc.get('source', f'Documento {i+1}')
                                st.markdown(f"**Fuente {i+1}:** `{source}`")
                                st.markdown(f"```\n{doc.get('content', 'Sin contenido')}\n```")
                    
                    # Si hay citas, mostrarlas
                    if "citations" in message and message["citations"]:
                        with st.expander("Ver referencias"):
                            for i, citation in enumerate(message["citations"]):
                                st.markdown(f'<div id="cita-{i+1}"></div>', unsafe_allow_html=True)
                                st.markdown(f"**[{i+1}]** `{citation['document_title']}`")
                                st.markdown(f"*\"{citation['cited_text']}\"*")
                    
                    # Si hay un flujo, mostrarlo
                    if "debug_info" in message:
                        with st.expander("Ver flujo de procesamiento"):
                            st.json(message["debug_info"])
                    
                    # Si hay subpreguntas, mostrarlas
                    if "subquestions" in message:
                        with st.expander("Ver subpreguntas generadas"):
                            for i, subq in enumerate(message["subquestions"]):
                                st.markdown(f"**Subpregunta {i+1}:** {subq['text']}")
                                if "answer" in subq and subq["answer"]:
                                    st.markdown(f"*Respuesta:* {subq['answer']}")
            
            # Input para la consulta
            query = st.chat_input("Escribe tu consulta sobre Renta para generar un concepto estructurado...")
            
            # Procesar la consulta
            if query:
                # Agregar la consulta del usuario a los mensajes
                st.session_state.renta_exp_messages.append({
                    "role": "user", 
                    "content": query
                })
                
                # Mostrar la consulta en la interfaz
                with st.chat_message("user"):
                    st.markdown(query)
                
                # Mostrar un spinner mientras se procesa la consulta
                with st.chat_message("assistant"):
                    # Crear un placeholder para mostrar el progreso
                    flow_placeholder = st.empty()
                    
                    # Usar una lista para almacenar los pasos del flujo (evita problemas con nonlocal)
                    flow_steps = []
                    
                    # Función para actualizar el flujo
                    def update_flow(step):
                        flow_steps.append(f"- {step}")
                        flow_text = ""
                        for s in flow_steps:
                            flow_text += s + "\n"
                        flow_placeholder.markdown(f"**Procesando:**\n{flow_text}")
                    
                    # Mostrar el flujo de procesamiento
                    update_flow(f"🔄 Iniciando procesamiento de la consulta sobre Renta...")
                    time.sleep(0.5)
                    
                    update_flow("🧠 Descomponiendo la consulta en subpreguntas...")
                    time.sleep(0.5)
                    
                    update_flow("🔍 Buscando documentos relevantes en Pinecone...")
                    time.sleep(1)
                    
                    try:
                        # Procesar con el flujo experimental
                        resultado = process_with_experimental_workflow(query, "Renta")
                        
                        # Verificar si se encontraron documentos
                        if not resultado.get("all_documents"):
                            update_flow("❌ No se encontraron documentos relevantes en Pinecone")
                            response = "Lo siento, no encontré información relevante sobre tu consulta en la base de conocimiento de Renta. Por favor, intenta reformular tu pregunta o consulta otra base de conocimiento."
                            final_flow = '\n'.join(flow_steps)
                            flow_placeholder.empty()
                            st.markdown(response)
                        else:
                            update_flow(f"📝 Encontrados {len(resultado.get('all_documents', []))} documentos relevantes")
                            time.sleep(0.5)
                            
                            update_flow("✍️ Respondiendo subpreguntas...")
                            time.sleep(0.5)
                            
                            update_flow("📊 Sintetizando respuesta final...")
                            time.sleep(0.5)
                            
                            update_flow("🔎 Revisando y refinando el concepto jurídico...")
                            time.sleep(0.5)
                            
                            if resultado.get("citations"):
                                update_flow(f"📌 Añadiendo {len(resultado.get('citations', []))} citas al concepto...")
                                time.sleep(0.5)
                            
                            update_flow("✨ Concepto jurídico generado con éxito!")
                            
                            # Guardar el flujo para mostrarlo en el historial
                            final_flow = '\n'.join(flow_steps)
                            
                            # Limpiar el placeholder
                            flow_placeholder.empty()
                            
                            # Obtener la respuesta final
                            if "reviewed_answer" in resultado and resultado["reviewed_answer"]:
                                respuesta = resultado["reviewed_answer"]
                            elif "final_answer" in resultado and resultado["final_answer"]:
                                respuesta = resultado["final_answer"]
                            else:
                                respuesta = "No se pudo generar una respuesta. Por favor, intente con otra consulta."
                            
                            # Formatear la respuesta con citas si existen
                            citas = resultado.get("citations", [])
                            if citas:
                                formatted_response = formatear_texto_con_citas(respuesta, citas)
                                st.markdown(formatted_response, unsafe_allow_html=True)
                            else:
                                st.markdown(respuesta)
                            
                            # Mostrar las citas si existen
                            if citas:
                                with st.expander("Ver referencias"):
                                    for i, citation in enumerate(citas):
                                        st.markdown(f'<div id="cita-{i+1}"></div>', unsafe_allow_html=True)
                                        st.markdown(f"**[{i+1}]** `{citation['document_title']}`")
                                        st.markdown(f"*\"{citation['cited_text']}\"*")
                    except Exception as e:
                        update_flow(f"❌ Error al generar respuesta: {str(e)}")
                        respuesta = f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
                        final_flow = '\n'.join(flow_steps)
                        flow_placeholder.empty()
                        st.markdown(respuesta)
                        citas = []
                        resultado = {"all_documents": [], "subquestions": [], "debug_info": {"error": str(e)}}
                
                # Mostrar las fuentes utilizadas
                with st.expander("Ver fuentes utilizadas"):
                    for i, doc in enumerate(resultado.get("all_documents", [])):
                        source = doc.get('source', f'Documento {i+1}')
                        st.markdown(f"**Fuente {i+1}:** `{source}`")
                        st.markdown(f"```\n{doc.get('content', 'Sin contenido')}\n```")
                
                # Mostrar el flujo de procesamiento
                with st.expander("Ver flujo de procesamiento"):
                    st.markdown(final_flow)
                
                # Mostrar subpreguntas generadas
                if "subquestions" in resultado:
                    with st.expander("Ver subpreguntas generadas"):
                        for i, subq in enumerate(resultado["subquestions"]):
                            st.markdown(f"**Subpregunta {i+1}:** {subq['text']}")
                            if "answer" in subq and subq["answer"]:
                                st.markdown(f"*Respuesta:* {subq['answer']}")
                
                # Agregar la respuesta del asistente a los mensajes
                st.session_state.renta_exp_messages.append({
                    "role": "assistant", 
                    "content": respuesta,
                    "documents": resultado.get("all_documents", []),
                    "citations": resultado.get("citations", []),
                    "subquestions": resultado.get("subquestions", []),
                    "debug_info": resultado.get("debug_info", {}),
                    "flow": final_flow
                })
except Exception as e:
    st.error(f"Error al inicializar Pinecone: {str(e)}")
    st.info("Por favor, asegúrate de que las credenciales de Pinecone estén correctamente configuradas en el archivo .env.") 