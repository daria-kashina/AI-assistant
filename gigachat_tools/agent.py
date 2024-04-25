import os
from dotenv import load_dotenv
from langchain_community.chat_models import GigaChat
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA

from langchain.prompts import load_prompt
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory

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
        self.system_prompt = "Ты ИИ-ассистент, ты помогаешь в научной деятельности."
        self.memory = ConversationBufferMemory(memory_key="history",input_key="query",output_key='answer',return_messages=True)


    def response_to_user_request(self, user_id: int, user_text_request: str) -> str:
        vectordb = Chroma(persist_directory="./data", embedding_function=self.embeddings)

        # template = """Используйте следующие фрагменты контекста, чтобы ответить на вопрос в конце.
        # Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумать ответ.
        # Cтарайтесь отвечать кратко по необходимости.
        # {context}
        # Вопрос: {query}
        # Вспомогательный вопрос: {answer}"""

        template = self.system_prompt + """ Используйте следующие фрагменты контекста, чтобы ответить на вопрос в конце.
        Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумать ответ.
        Cтарайтесь отвечать кратко по необходимости.
        Вопрос: {query}. {context}"""

        # qa_chain_prompt = PromptTemplate(input_variables=["context", "query", "answer"], template=template)
        qa_chain_prompt = PromptTemplate(input_variables=["query", "context"], template=template)

        answer = """Если задан вопрос, постарайся на него ответить. Если вопрос не был задан постарайся 
            найти похожую информацию, статьи с похожей тематикой и выведи информацию о них."""

        # format the prompt to add variable values
        # prompt_formatted_str = qa_chain_prompt.format(
        #     context = self.system_prompt,
        #     query = user_text_request,
        #     answer = """Если задан вопрос, постарайся на него ответить. Если вопрос не был задан постарайся 
        #     найти похожую информацию, статьи с похожей тематикой и выведи информацию о них."""
        #     )

        # print("ок1")
        # print(prompt_formatted_str)
        # print(type(prompt_formatted_str))

        print(type(qa_chain_prompt))

        chain_type_kwargs = {"verbose": True, "prompt": qa_chain_prompt, "memory": self.memory}
        chain_type_kwargs = {"verbose": True,  "memory": self.memory}

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            # chain_type='map_rerank',
            retriever=vectordb.as_retriever(search_kwargs={'k': 10}),
            chain_type_kwargs=chain_type_kwargs,
            return_source_documents=True)
        print("ок2")

        # prompt = load_prompt("lc://prompts/content/spell_correction.yaml")
        # chain = prompt | self.llm

        # result = qa_chain.invoke({'query': user_text_request})
        # result = qa_chain.invoke({'query': user_text_request, 'context':self.system_prompt, 'answer': answer})
        # result = qa_chain.invoke({'query': user_text_request})
        result = qa_chain.invoke({'query': user_text_request, 'context':self.system_prompt})

        print(result)
        return result['result']
