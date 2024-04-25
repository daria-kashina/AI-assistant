from chromadb.config import Settings
import codecs
from config import cred

from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings

llm = GigaChat(credentials=cred, verify_ssl_certs=False, streaming=True, scope='GIGACHAT_API_CORP')

# Изменить encoding при необходимости
loader = TextLoader(
    "/Users/darakasina/Desktop/DS/deepagents_hackaton/AI-assistant/example2.txt", 
    encoding='utf-8'
    )
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)
print(documents)
documents = text_splitter.split_documents(documents)
print(f"Total documents: {len(documents)}")



embeddings = GigaChatEmbeddings(
    credentials=cred, verify_ssl_certs=False, scope='GIGACHAT_API_CORP'
)

db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
    persist_directory='/Users/darakasina/Desktop/DS/deepagents_hackaton/AI-assistant/db'
)

db.persist()

print("База данных успешно создана и сохранена.")
print(f"Место нахождения базы данных: {db._persist_directory}")
