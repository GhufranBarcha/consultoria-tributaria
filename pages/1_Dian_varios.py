import streamlit as st
from dotenv import load_dotenv
import os
import time
import re
from graph.chains.retrieval import chroma_retriever
from graph.chains.openai_generation import generate_with_openai

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Dian varios",
    page_icon="💰",
    layout="wide"
)

# Título de la página
st.title("Consulta sobre diversos conceptos de la Dian")

# Descripción de la página
st.markdown("""
Esta sección le permite realizar consultas sobre diversos conceptos de la DIAN. Hay 3057 documentos sobre aduanero, cambiario, consiumo, carbono, saludables, IVA, renta, retención y timbre. Conceptos desde enero de 2017 a diciembre de 2024
""")

# Verificar si la colección existe
chroma_dir = "./.chroma"
if not os.path.exists(chroma_dir):
    st.warning("La base de datos de IVA no está inicializada. Por favor, ejecuta el script de ingesta primero.")
else:
    # Inicializar estado de sesión para IVA
    if "iva_messages" not in st.session_state:
        st.session_state.iva_messages = []
    
    # Función para formatear el texto con citas numeradas
    def formatear_texto_con_citas(texto, citas):
        """
        Formatea el texto con citas numeradas en HTML.
        """
        if not citas:
            return texto
        
        # Reemplazar los corchetes de cita por etiquetas HTML para mejorar la visualización
        texto_formateado = re.sub(r'\[(\d+)\]', r'<sup>[\1]</sup>', texto)
        
        return texto_formateado
    
    # Mostrar mensajes anteriores
    for message in st.session_state.iva_messages:
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
    query = st.chat_input("Escribe tu consulta sobre diversos temas...")
    
    # Procesar la consulta
    if query:
        # Agregar la consulta del usuario a los mensajes
        st.session_state.iva_messages.append({
            "role": "user", 
            "content": query
        })
        
        # Mostrar la consulta en la interfaz
        with st.chat_message("user"):
            st.markdown(query)
        
        # Mostrar un spinner mientras se procesa la consulta
        with st.chat_message("assistant"):
            # Crear un placeholder para mostrar el flujo en tiempo real
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
            update_flow(f"🔄 Iniciando procesamiento de la consulta...")
            time.sleep(0.5)
            
            update_flow("🧠 Analizando la consulta...")
            time.sleep(0.5)
            
            update_flow("🔍 Buscando documentos relevantes en Chroma...")
            time.sleep(1)
            
            # CAMBIO IMPORTANTE: Consultar directamente a Chroma en lugar de usar el grafo
            print("IVA.py: Consultando directamente a Chroma")
            documents = chroma_retriever.invoke(query)
            print(f"IVA.py: Recuperados {len(documents)} documentos de Chroma")
            
            # Verificar si se encontraron documentos
            if not documents:
                update_flow("❌ No se encontraron documentos relevantes en Chroma")
                response = "Lo siento, no encontré información relevante sobre tu consulta en la base de conocimiento de IVA. Por favor, intenta reformular tu pregunta o consulta otra base de conocimiento."
                final_flow = '\n'.join(flow_steps)
                flow_placeholder.empty()
                st.markdown(response)
            else:
                update_flow(f"📝 Encontrados {len(documents)} documentos relevantes")
                time.sleep(0.5)
                
                update_flow("✍️ Generando respuesta...")
                time.sleep(0.5)
                
                # Generar respuesta con OpenAI
                try:
                    openai_response = generate_with_openai(query, documents)
                    response = openai_response["text"]
                    citations = openai_response.get("citations", [])
                    
                    update_flow("🔎 Verificando que no haya alucinaciones...")
                    time.sleep(0.5)
                    
                    update_flow("✅ Verificando que la respuesta aborde la consulta...")
                    time.sleep(0.5)
                    
                    if citations:
                        update_flow(f"📌 Añadiendo {len(citations)} citas a la respuesta...")
                        time.sleep(0.5)
                    
                    update_flow("✨ Respuesta generada con éxito!")
                    
                    # Guardar el flujo para mostrarlo en el historial
                    final_flow = '\n'.join(flow_steps)
                    
                    # Limpiar el placeholder
                    flow_placeholder.empty()
                    
                    # Formatear la respuesta con citas si existen
                    if citations:
                        formatted_response = formatear_texto_con_citas(response, citations)
                        st.markdown(formatted_response, unsafe_allow_html=True)
                    else:
                        st.markdown(response)
                    
                    # Mostrar las citas si existen
                    if citations:
                        with st.expander("Ver referencias"):
                            for i, citation in enumerate(citations):
                                st.markdown(f"**[{i+1}]** `{citation['document_title']}`")
                                st.markdown(f"*\"{citation['cited_text']}\"*")
                except Exception as e:
                    update_flow(f"❌ Error al generar respuesta: {str(e)}")
                    response = f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}"
                    final_flow = '\n'.join(flow_steps)
                    flow_placeholder.empty()
                    st.markdown(response)
                    citations = []
            
            # Mostrar las fuentes utilizadas
            with st.expander("Ver fuentes utilizadas"):
                for i, doc in enumerate(documents):
                    source = doc.metadata.get('source', f'Documento {i+1}')
                    st.markdown(f"**Fuente {i+1}:** `{source}`")
                    st.markdown(f"```\n{doc.page_content}\n```")
            
            # Mostrar el flujo de procesamiento
            with st.expander("Ver flujo de procesamiento"):
                st.markdown(final_flow)
        
        # Agregar la respuesta del asistente a los mensajes
        st.session_state.iva_messages.append({
            "role": "assistant", 
            "content": response,
            "documents": documents,
            "flow": final_flow,
            "citations": citations if 'citations' in locals() else []
        })

# Información adicional en el sidebar
with st.sidebar:
    st.header("Información sobre diversos temas")
    
    # Botón para limpiar la conversación
    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.iva_messages = []
        st.rerun()