from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
import os
import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener la clave API de OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    # Intentar obtener la clave de los secretos de Streamlit
    try:
        openai_api_key = st.secrets["OPENAI_API_KEY"]
    except:
        raise ValueError("No se encontró la clave API de OpenAI. Por favor, configúrela en las variables de entorno o en los secretos de Streamlit.")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
