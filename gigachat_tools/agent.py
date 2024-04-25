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

# Создаем словарь для хранения контекста диалога
dialog_context = {}

def is_retrievable_from_database(prompt):
    # Проводим проверку, доступен ли ответ на данный запрос в базе данных
    # Вернуть True, если ответ доступен в базе данных, и False в противном случае
    if "запрос" in prompt: 
        return True
    else:
        return False

async def context_aware_agent(prompt, credentials, scope):
    # Получаем предыдущий диалог из контекста
    previous_dialog = dialog_context.get('previous_dialog')
    previous_prompt = previous_dialog['prompt'] if previous_dialog else None
    previous_response = previous_dialog['response'] if previous_dialog else None
    # Добавляем предыдущий диалог к текущему запросу
    full_prompt = f"{previous_prompt} {previous_response}" if previous_prompt else prompt

    if is_retrievable_from_database(prompt):  # Функция для проверки, можно ли получить ответ из базы данных
        result = retrieve_from_database(full_prompt)  # Функция для получения ответа из базы данных
    else:
        embeddings = GigaChatEmbeddings(
            credentials=os.environ['GIGACHAT_CRED'], 
            verify_ssl_certs=False, scope='GIGACHAT_API_CORP'
        )

        vectordb = Chroma(persist_directory="./data", embedding_function=embeddings)

        llm = GigaChat(credentials=os.environ['GIGACHAT_CRED'], 
        verify_ssl_certs=False, scope='GIGACHAT_API_CORP')

        query = {"query": full_prompt} #NEW

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            # retriever=ensemble_retriever,
            retriever=vectordb.as_retriever(search_kwargs={'k': 18}),
            return_source_documents=True,
        )

        # Теперь можно выполнять запросы к нашей цепочке вопросов и ответов
        result = qa_chain.invoke({'query': full_prompt})

    # print(result)

    # Сохраняем текущий диалог в контекст
    dialog_context['previous_dialog'] = {'prompt': prompt, 'response': result['result']}

    return result['result']   
