from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

def definir_prompt(llm_gpt, assuntos):

    prompt = ChatPromptTemplate.from_template("""
            {context} 
            {input}
            """)
    input = """## Objetivo
                ## Siga estas diretrizes:
                0. Crie um debate com base nos candidatos a presidência dos EUA de 2024.
                1. Faça em português, Brail.
                2. Coloque questões de {assuntos}.
                ## Depois de gerar o debate:
                - Resuma a ideia central de cada candidato.
                """.format(assuntos=assuntos)

    document_chain = create_stuff_documents_chain(llm_gpt, prompt)

    return document_chain, input