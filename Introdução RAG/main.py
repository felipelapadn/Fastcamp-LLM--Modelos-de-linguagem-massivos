from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from rag import definir_prompt
from vectordb import url_to_vector
from model_initializers import initialize_gpt4o

load_dotenv()
llm_gpt = initialize_gpt4o()
retrieval = url_to_vector()
document_chain, input_ = definir_prompt(llm_gpt, ["sustentabilidade", "imigracoes"])
retrieval_chain = create_retrieval_chain(retrieval, document_chain)
response = retrieval_chain.invoke({"input": input_})
print(response["answer"])

