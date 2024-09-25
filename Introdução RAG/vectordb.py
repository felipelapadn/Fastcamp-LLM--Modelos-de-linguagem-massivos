import json
import logging
import random
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def url_to_vector() -> any:

    loader = WebBaseLoader("https://www.cnnbrasil.com.br/internacional/quem-sao-os-candidatos-a-presidencia-nas-eleicoes-dos-eua-de-2024/")
    dados_embedding = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=20) # chunk_size=5000, chunk_overlap=10
    documents = text_splitter.split_documents(dados_embedding)
    vector_database = FAISS.from_documents(documents, embeddings)
    retrieval = vector_database.as_retriever()

    return retrieval