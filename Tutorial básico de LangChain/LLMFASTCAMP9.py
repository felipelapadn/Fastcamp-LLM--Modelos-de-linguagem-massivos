from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

def generate_nomes_curso(plataforma):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6)
    cursos_template = PromptTemplate(input_variables=["plataforma"],
                                     template="Produza 5 nomes de cursos para"
                                              "gerencia na plataforma {plataforma}.")
    chain = LLMChain(llm=llm, prompt=cursos_template)
    res = chain({"plataforma": plataforma})
    return res

if __name__=="__main__":
    print(generate_nomes_curso("Udemy"))