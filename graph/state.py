from typing import List, TypedDict, Dict, Any, Optional


class GraphState(TypedDict, total=False):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        citations: optional list of citations from Claude
        topic: optional topic for the query (e.g., "IVA", "Renta")
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]
    citations: Optional[List[Dict[str, Any]]]
    topic: Optional[str]
