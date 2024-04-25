import os
from dotenv import load_dotenv
from langchain_community.chat_models import GigaChat
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA

# Read .env file
load_dotenv()


# async def simple_agent(credentials, scope):
#     embeddings = GigaChatEmbeddings(
#         credentials=credentials,
#         scope=scope,
#         verify_ssl_certs=False)
#     result = embeddings.embed_documents(texts=["Привет!"])
#     # print(result)
#     return "Проверка, все хорошо gigachat работает!"



async def simple_agent(text, credentials, scope):
    embeddings = GigaChatEmbeddings(
        credentials=os.environ['GIGACHAT_CRED'], 
        verify_ssl_certs=False, scope='GIGACHAT_API_CORP'
    )

    vectordb = Chroma(persist_directory="./data", embedding_function=embeddings)

    llm = GigaChat(credentials=os.environ['GIGACHAT_CRED'], 
    verify_ssl_certs=False, scope='GIGACHAT_API_CORP')

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # retriever=ensemble_retriever,
        retriever=vectordb.as_retriever(search_kwargs={'k': 18}),
        return_source_documents=True,
    )

    # Теперь можно выполнять запросы к нашей цепочке вопросов и ответов
    result = qa_chain.invoke({'query': text})
    # print(result)

    return result['result']


class ScientificAIAgent:
    def __init__(self):
        self.credentials = os.environ['GIGACHAT_CRED']
        self.embeddings = GigaChatEmbeddings(
            credentials = self.credentials, 
            verify_ssl_certs = False, 
            scope = 'GIGACHAT_API_CORP')
        self.llm = GigaChat(credentials = self.credentials, 
            verify_ssl_certs = False, 
            scope = 'GIGACHAT_API_CORP')

    def response_to_user_request(self, user_id: int, user_text_request: str) -> str:
        vectordb = Chroma(persist_directory="./data", embedding_function=self.embeddings)

        qa_chain = RetrievalQA.from_chain_type(
        llm=self.llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={'k': 10}),
        return_source_documents=True)
        result = qa_chain.invoke({'query': user_text_request})
        return result['result']
