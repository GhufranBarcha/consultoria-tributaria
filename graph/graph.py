from dotenv import load_dotenv

from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

# Variable global para controlar la depuración
DEBUG = False

def set_debug(debug_mode):
    global DEBUG
    DEBUG = debug_mode

def debug_print(message):
    if DEBUG:
        print(message)

def decide_to_generate(state):
    debug_print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        debug_print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        debug_print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    debug_print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Verificar si hay citas en el estado
    if "citations" in state and state["citations"]:
        print(f"grade_generation: Se encontraron {len(state['citations'])} citas en el estado")
        # Imprimir las primeras citas para depuración
        for i, citation in enumerate(state['citations'][:2]):
            print(f"grade_generation: Cita {i+1}: {citation.get('document_title', 'Sin título')}")
    else:
        print("grade_generation: NO SE ENCONTRARON CITAS EN EL ESTADO")
    
    # Verificar si la respuesta tiene la estructura esperada
    if "has_structure" in state and state["has_structure"]:
        print("grade_generation: La respuesta tiene la estructura esperada")
    else:
        print("grade_generation: La respuesta NO tiene la estructura esperada")

    # Imprimir las primeras 100 caracteres de la respuesta para depuración
    print(f"grade_generation: Primeros 100 caracteres de la respuesta: {generation[:100]}...")

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        debug_print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        debug_print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            debug_print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            debug_print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        debug_print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    debug_print("---ROUTE QUESTION---")
    question = state["question"]
    
    # Verificar si hay un tema específico en el estado
    if "topic" in state:
        topic = state["topic"]
        debug_print(f"---TOPIC FOUND IN STATE: {topic}---")
        # Si el tema es Renta, ir directamente a recuperación
        if topic == "Renta":
            debug_print("---ROUTE QUESTION TO RAG (RENTA)---")
            return RETRIEVE
    
    # Comportamiento normal para otros casos
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        debug_print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        debug_print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")