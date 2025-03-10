from dotenv import load_dotenv

load_dotenv()

# Importar el grafo
from graph.graph import app

print("Generando el grafo...")
# El código para generar el archivo Mermaid ya está en graph.py
# Solo necesitamos importar el módulo para que se ejecute

print("Grafo generado. El código Mermaid se ha guardado en 'graph.mmd'")
print("Para visualizar el grafo, puede usar una herramienta en línea como https://mermaid.live/")
print("o instalar la extensión Mermaid para VSCode y abrir el archivo graph.mmd") 