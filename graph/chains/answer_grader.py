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


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
