from langchain.chat_models.gigachat import GigaChat
CREDENTIALS = 'Yjg4MTQzMmUtNDAwMS00NDk0LThjOGUtNmU5ZWQ2YzQ4NDQ2OmQ4MWMxZGZiLTFmNGYtNDk5NS05OGQzLTBiMzYyYWJmNjk3OA=='

llm = GigaChat(credentials=CREDENTIALS, verify_ssl_certs=False, streaming=True)
from langchain.schema import HumanMessage


import codecs
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

loader = TextLoader("C:\\WebProject\\AI\\AI-assistant\\giga_env\\resources\\example2.txt")
documents = loader.load()

#with codecs.open("C:\\WebProject\\AI\\AI-assistant\\giga_env\\resources\\example.txt", "r", encoding="utf-8") as f:
#    text = f.read()
#    loader = TextLoader(text)
#    documents = loader.load()
#print(documents)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)
print(documents)
documents = text_splitter.split_documents(documents)
print(f"Total documents: {len(documents)}")

from chromadb.config import Settings
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings

embeddings = GigaChatEmbeddings(
    credentials=CREDENTIALS, verify_ssl_certs=False
)

db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
    persist_directory='./data'
)

db.persist()

print(db)