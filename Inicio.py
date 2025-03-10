import streamlit as st
from dotenv import load_dotenv
import os
from graph.graph import app, set_debug
import time
import re

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Asistente Jurídico Tributario",
    page_icon="⚖️",
    layout="wide"
)

# Título de la página en la barra lateral (esto cambiará el nombre de la pestaña)
st.sidebar.title("")

# Desactivar la depuración en consola
set_debug(True)

# Función para obtener la colección adecuada según la subárea
def obtener_coleccion(subarea=None):
    # Para mantener compatibilidad con la estructura actual
    if subarea == "Dian varios":
        return "legal-docs-chroma"  # Usa la colección existente
    
    # Para nuevas subáreas (futuras implementaciones)
    if subarea:
        return f"derecho_tributario-{subarea.lower().replace(' ', '_')}-chroma"
    else:
        return "derecho_tributario-chroma"

# Verificar si una colección existe
def verificar_coleccion(subarea=None):
    collection_name = obtener_coleccion(subarea)
    chroma_dir = "./.chroma"
    
    # Verificar si la carpeta .chroma existe
    if not os.path.exists(chroma_dir):
        return False
    
    # Caso especial para Renta (usa Pinecone)
    if subarea == "Renta":
        print("verificar_coleccion: Verificando Pinecone para Renta")
        # Siempre devolver True para Renta, ya que forzaremos el uso de Pinecone
        return True
    
    # Intentar acceder a la colección (implementación básica)
    # En una implementación completa, verificaríamos si la colección existe en Chroma
    if collection_name == "legal-docs-chroma":
        return True  # Asumimos que la colección principal existe
    
    # Para otras colecciones, verificar si ya se han creado (futuras implementaciones)
    return False

# Función para formatear el texto con citas numeradas
def formatear_texto_con_citas(texto, citas):
    """
    Ya no necesitamos formatear el texto con citas, ya que Claude las incluye directamente.
    Esta función ahora solo se asegura de que las citas se muestren correctamente en HTML.
    """
    if not citas:
        return texto
    
    # Reemplazar los corchetes de cita por etiquetas HTML para mejorar la visualización
    texto_formateado = re.sub(r'\[(\d+)\]', r'<sup>[\1]</sup>', texto)
    
    return texto_formateado

# Inicializar estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

# Obtener parámetros de URL para mantener el estado entre recargas
query_params = st.query_params
default_subarea = query_params.get("subarea", ["IVA"])[0] if "subarea" in query_params else "IVA"

# Verificar que la subárea exista en nuestras opciones
if default_subarea not in ["Dian varios", "Renta"]:
    default_subarea = "Dian varios"

# Inicializar o actualizar el estado con los valores de URL
if "subarea_seleccionada" not in st.session_state:
    st.session_state.subarea_seleccionada = default_subarea

# Título principal
st.title("⚖️ Asistente Jurídico Tributario")

# Tabs para las subáreas
tab1, tab2 = st.tabs(["Dian varios", "Renta"])

with tab1:
    if st.button("Seleccionar Dian varios", key="select_dian_varios", use_container_width=True, 
               type="primary" if st.session_state.subarea_seleccionada == "Dian varios" else "secondary"):
        st.session_state.subarea_seleccionada = "Dian varios"
        st.query_params.update(subarea="Dian varios")
        st.rerun()

with tab2:
    if st.button("Seleccionar Renta", key="select_renta", use_container_width=True,
               type="primary" if st.session_state.subarea_seleccionada == "Renta" else "secondary"):
        st.session_state.subarea_seleccionada = "Renta"
        st.query_params.update(subarea="Renta")
        st.rerun()

# Descripción de la aplicación
st.markdown(f"""
## Asistente de Derecho Tributario

Esta aplicación utiliza técnicas avanzadas de Retrieval Augmented Generation (RAG) para proporcionar 
respuestas precisas a consultas jurídicas en el área de derecho tributario colombiano.

### Base de conocimiento actual: {st.session_state.subarea_seleccionada}

- **Dian varios**: Consultas sobre conceptos varios de la Dian desde enero de 2017 hasta diciembre de 2024 (Chroma)
- **Renta**: Consultas sobre el Impuesto de Renta desde enero de 2017 hasta diciembre de 2024 (Pinecone)
""")

# Verificar si la colección existe
collection_name = obtener_coleccion(st.session_state.subarea_seleccionada)
if not verificar_coleccion(st.session_state.subarea_seleccionada):
    st.warning(f"La base de datos para {st.session_state.subarea_seleccionada} no está inicializada. Por favor, ejecuta el script de ingesta primero.")
else:
    # Filtrar mensajes por subárea
    area_messages = [m for m in st.session_state.messages 
                   if m.get("subarea") == st.session_state.subarea_seleccionada]
    
    # Mostrar mensajes anteriores
    for message in area_messages:
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
                        source = doc.metadata.get('source', f'Documento {i+1}')
                        st.markdown(f"**Fuente {i+1}:** `{source}`")
                        st.markdown(f"```\n{doc.page_content}\n```")
            
            # Si hay citas, mostrarlas
            if "citations" in message and message["citations"]:
                with st.expander("Ver referencias"):
                    for i, citation in enumerate(message["citations"]):
                        st.markdown(f"**[{i+1}]** `{citation['document_title']}`")
                        st.markdown(f"*\"{citation['cited_text']}\"*")
            
            # Si hay un flujo, mostrarlo
            if "flow" in message:
                with st.expander("Ver flujo de procesamiento"):
                    st.markdown(message["flow"])
    
    # Input para la consulta
    query = st.chat_input(f"Escribe tu consulta sobre {st.session_state.subarea_seleccionada}...")
    
    # Procesar la consulta
    if query:
        # Agregar la consulta del usuario a los mensajes
        st.session_state.messages.append({
            "role": "user", 
            "content": query,
            "subarea": st.session_state.subarea_seleccionada
        })
        
        # Mostrar la consulta en la interfaz
        with st.chat_message("user"):
            st.markdown(query)
        
        # Mostrar un spinner mientras se procesa la consulta
        with st.chat_message("assistant"):
            # Crear un placeholder para mostrar el flujo en tiempo real
            flow_placeholder = st.empty()
            
            # Inicializar el flujo
            flow_steps = []
            
            # Función para actualizar el flujo
            def update_flow(step):
                flow_steps.append(step)
                flow_text = ""
                for s in flow_steps:
                    flow_text += f"- {s}\n"
                flow_placeholder.markdown(f"**Procesando:**\n{flow_text}")
            
            # Mostrar el flujo de procesamiento
            update_flow(f"🔄 Iniciando procesamiento de la consulta en {st.session_state.subarea_seleccionada}...")
            time.sleep(0.5)
            
            # Invocar el grafo con la consulta
            update_flow("🔍 Buscando documentos relevantes...")
            result = app.invoke(input={"question": query, "topic": st.session_state.subarea_seleccionada})
            
            # Extraer la respuesta y los documentos
            response = result.get("generation", "No se pudo generar una respuesta.")
            documents = result.get("documents", [])
            
            # Verificar si hay citas en el resultado
            if "citations" in result:
                citations = result.get("citations", [])
                update_flow(f"📌 Se encontraron {len(citations)} citas en el resultado.")
            else:
                # Si no hay citas en el resultado o la respuesta no tiene la estructura esperada
                if "REFERENCIA" not in response or "ANÁLISIS" not in response:
                    update_flow("🔄 La respuesta no tiene la estructura esperada. Generando respuesta estructurada con OpenAI...")
                    try:
                        from graph.chains.openai_generation import generate_with_openai
                        # Importante: Usar los documentos que ya se recuperaron
                        # No volver a consultar otra fuente
                        openai_response = generate_with_openai(query, documents)
                        response = openai_response["text"]
                        citations = openai_response.get("citations", [])
                        update_flow(f"✅ Respuesta estructurada generada con {len(citations)} citas.")
                    except Exception as e:
                        update_flow(f"❌ Error al generar respuesta estructurada: {str(e)}")
                        citations = []
                else:
                    # Si no hay citas en el resultado, intentar extraerlas de la respuesta
                    citations = []
            
            # Mostrar la respuesta
            if not documents:
                update_flow("❌ No se encontraron documentos relevantes.")
                st.markdown("Lo siento, no encontré información relevante sobre tu consulta en la base de conocimiento de " + 
                          f"{st.session_state.subarea_seleccionada}. Por favor, intenta reformular tu pregunta o consulta otra base de conocimiento.")
            else:
                update_flow("✅ Documentos relevantes encontrados.")
                update_flow("🤖 Generando respuesta...")
                
                # Si hay citas, formatear el texto con ellas
                if citations:
                    formatted_response = formatear_texto_con_citas(response, citations)
                    st.markdown(formatted_response, unsafe_allow_html=True)
                else:
                    st.markdown(response)
                
                # Mostrar documentos
                with st.expander("Ver fuentes utilizadas"):
                    for i, doc in enumerate(documents):
                        source = doc.metadata.get('source', f'Documento {i+1}')
                        st.markdown(f"**Fuente {i+1}:** `{source}`")
                        st.markdown(f"```\n{doc.page_content}\n```")
                
                # Mostrar citas
                if citations:
                    with st.expander("Ver referencias"):
                        for i, citation in enumerate(citations):
                            st.markdown(f"**[{i+1}]** `{citation['document_title']}`")
                            st.markdown(f"*\"{citation['cited_text']}\"*")
            
            # Guardar el flujo final
            final_flow = ""
            for s in flow_steps:
                final_flow += f"- {s}\n"
            
            # Agregar la respuesta a los mensajes
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "documents": documents,
                "flow": final_flow,
                "subarea": st.session_state.subarea_seleccionada,
                "citations": citations if citations else []
            })