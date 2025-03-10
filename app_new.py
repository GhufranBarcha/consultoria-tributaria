import streamlit as st
from dotenv import load_dotenv
import os
from graph.graph import app, set_debug
import time
import subprocess
import re

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Asistente Jurídico con RAG Avanzado",
    page_icon="⚖️",
    layout="wide"
)

# Desactivar la depuración en consola
set_debug(True)

# Definir áreas y subáreas del derecho
AREAS_DERECHO = {
    "Derecho tributario": ["IVA", "Renta", "GMF", "Procedimientos"],
    "Derecho societario": ["Sociedades", "Contratos", "Inversión extranjera"],
    "Derecho penal": ["Delitos", "Procedimiento penal"]
}

# Función para obtener la colección adecuada según el área y subárea
def obtener_coleccion(area, subarea=None):
    # Para mantener compatibilidad con la estructura actual
    if area == "Derecho tributario" and subarea == "IVA":
        return "legal-docs-chroma"  # Usa la colección existente
    
    # Para nuevas áreas y subáreas (futuras implementaciones)
    if subarea:
        return f"{area.lower().replace(' ', '_')}-{subarea.lower().replace(' ', '_')}-chroma"
    else:
        return f"{area.lower().replace(' ', '_')}-chroma"

# Verificar si una colección existe
def verificar_coleccion(area, subarea=None):
    collection_name = obtener_coleccion(area, subarea)
    chroma_dir = "./.chroma"
    
    # Verificar si la carpeta .chroma existe
    if not os.path.exists(chroma_dir):
        return False
    
    # Caso especial para Renta (usa Pinecone)
    if area == "Derecho tributario" and subarea == "Renta":
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
default_area = query_params.get("area", ["Derecho tributario"])[0] if "area" in query_params else "Derecho tributario"
default_subarea = query_params.get("subarea", ["IVA"])[0] if "subarea" in query_params else "IVA"

# Verificar que el área y subárea existan en nuestras opciones
if default_area not in AREAS_DERECHO:
    default_area = "Derecho tributario"
if default_subarea not in AREAS_DERECHO.get(default_area, []):
    default_subarea = AREAS_DERECHO[default_area][0]

# Inicializar o actualizar el estado con los valores de URL
if "area_seleccionada" not in st.session_state:
    st.session_state.area_seleccionada = default_area
if "subarea_seleccionada" not in st.session_state:
    st.session_state.subarea_seleccionada = default_subarea

# Título principal
st.title("⚖️ Asistente Jurídico con RAG Avanzado")

# Sidebar para navegación
with st.sidebar:
    st.header("Navegación")
    
    # Derecho tributario
    with st.expander("Derecho tributario", expanded=True):
        # IVA
        if st.button("IVA", key="sidebar_iva", use_container_width=True, 
                   type="primary" if st.session_state.subarea_seleccionada == "IVA" else "secondary"):
            st.session_state.area_seleccionada = "Derecho tributario"
            st.session_state.subarea_seleccionada = "IVA"
            st.query_params.update(area="Derecho tributario", subarea="IVA")
            st.rerun()
        
        # Renta
        if st.button("Renta", key="sidebar_renta", use_container_width=True,
                   type="primary" if st.session_state.subarea_seleccionada == "Renta" else "secondary"):
            st.session_state.area_seleccionada = "Derecho tributario"
            st.session_state.subarea_seleccionada = "Renta"
            st.query_params.update(area="Derecho tributario", subarea="Renta")
            st.rerun()
        
        # GMF (próximamente)
        if st.button("GMF (próximamente)", key="sidebar_gmf", use_container_width=True, disabled=True):
            pass
        
        # Procedimientos (próximamente)
        if st.button("Procedimientos (próximamente)", key="sidebar_procedimientos", use_container_width=True, disabled=True):
            pass
    
    # Derecho societario (próximamente)
    with st.expander("Derecho societario (próximamente)", expanded=False):
        st.info("Esta sección estará disponible próximamente")
    
    # Derecho penal (próximamente)
    with st.expander("Derecho penal (próximamente)", expanded=False):
        st.info("Esta sección estará disponible próximamente")
    
    st.markdown("---")
    
    st.markdown("""
    ### Sobre este asistente
    
    Este asistente utiliza técnicas avanzadas de RAG para proporcionar respuestas precisas a consultas jurídicas.
    
    **Características:**
    - Búsqueda semántica en documentos legales
    - Evaluación de relevancia de documentos
    - Verificación de respuestas
    - Citas y referencias a fuentes
    """)

# Descripción de la aplicación
st.markdown("""
## Bienvenido al Asistente Jurídico con RAG Avanzado

Esta aplicación utiliza técnicas avanzadas de Retrieval Augmented Generation (RAG) para proporcionar 
respuestas precisas a consultas jurídicas en diferentes áreas del derecho tributario colombiano.

### Bases de conocimiento disponibles:

- **IVA**: Consultas sobre el Impuesto al Valor Agregado
- **Renta**: Consultas sobre el Impuesto de Renta
- **GMF**: Consultas sobre el Gravamen a los Movimientos Financieros (próximamente)
- **Procedimientos**: Consultas sobre procedimientos tributarios (próximamente)

### Cómo usar la aplicación:

1. Seleccione una de las bases de conocimiento en el menú lateral
2. Formule su consulta en el campo de texto
3. Reciba una respuesta detallada con citas a las fuentes relevantes
""")

# Mostrar información sobre las tecnologías utilizadas
st.markdown("""
### Tecnologías utilizadas:

- **LangGraph**: Para la orquestación del flujo de RAG
- **Chroma**: Base de vectores para IVA y otras colecciones
- **Pinecone**: Base de vectores para Renta
- **OpenAI**: Para embeddings y generación de respuestas
""")

# Mostrar el área y subárea seleccionada
st.subheader(f"Consultas de {st.session_state.subarea_seleccionada}")

# Verificar si la colección existe
collection_name = obtener_coleccion(st.session_state.area_seleccionada, st.session_state.subarea_seleccionada)
if not verificar_coleccion(st.session_state.area_seleccionada, st.session_state.subarea_seleccionada):
    if st.session_state.area_seleccionada == "Derecho tributario" and st.session_state.subarea_seleccionada == "IVA":
        st.warning("La base de datos principal no está inicializada. Por favor, ejecuta el script de ingesta primero.")
    else:
        st.info(f"La base de conocimiento para {st.session_state.subarea_seleccionada} aún no está disponible.")
else:
    # Filtrar mensajes por área y subárea
    area_messages = [m for m in st.session_state.messages 
                   if m.get("area") == st.session_state.area_seleccionada and 
                      m.get("subarea") == st.session_state.subarea_seleccionada]
    
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
            "area": st.session_state.area_seleccionada,
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
            citations = result.get("citations", [])
            
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
                    formatted_answer = formatear_texto_con_citas(response, citations)
                    st.markdown(formatted_answer, unsafe_allow_html=True)
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
                "area": st.session_state.area_seleccionada,
                "subarea": st.session_state.subarea_seleccionada,
                "citations": citations if citations else []
            }) 