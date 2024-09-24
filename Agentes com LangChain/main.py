from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import YouTubeSearchTool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub

load_dotenv()

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool_wik = WikipediaQueryRun(api_wrapper=api_wrapper)
tool_yt = YouTubeSearchTool()

# print(tool_wik.run("planet"))
tools = [tool_yt, tool_wik]
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6)
agente = create_openai_functions_agent(llm, tools, prompt)
agente_exec = AgentExecutor(agent=agente, tools=tools, verbose=True)
# agente_exec.invoke({"input": "Ola, tudo certinho?"})
agente_exec.invoke({"input": "Me de alguns links de videos do Youtube que fale sobre feijao"})