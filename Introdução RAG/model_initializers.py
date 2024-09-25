from langchain_openai import ChatOpenAI
from transformers import pipeline

def initialize_gpt4o():
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)