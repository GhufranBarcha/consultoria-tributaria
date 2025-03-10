from dotenv import load_dotenv

load_dotenv()

# Importar el grafo
from graph.graph import app

print("Generando el grafo en formato PNG...")
try:
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("Grafo generado exitosamente como 'graph.png'")
except Exception as e:
    print(f"Error al generar el grafo: {e}")
    print("Intentando método alternativo...")
    try:
        import os
        # Guardar el código Mermaid
        mermaid_code = app.get_graph().draw_mermaid()
        with open("graph.mmd", "w") as f:
            f.write(mermaid_code)
        print("Código Mermaid guardado en 'graph.mmd'")
        
        # Intentar convertir a PNG usando mmdc si está instalado
        os.system("mmdc -i graph.mmd -o graph.png")
        print("Verificar si se generó 'graph.png'")
    except Exception as e2:
        print(f"Error en método alternativo: {e2}")
        print("No se pudo generar el grafo en PNG.") 