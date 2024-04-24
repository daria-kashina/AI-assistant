from config import cred
from gigachain_community.chat_models import GigaChat
from gigachain.memory.chat_message_histories import FileChatMessageHistory
from gigachain_expiremental.autonomous_agents import AutoGPT

llm = GigaChat(
    verbose=True,
    temperature=0,
    model="... big giga model ...",
    credentials=cred,
    base_url=...,
    verify_ssl_certs=False,
    timeout=300,
    scope='GIGACHAT_API_CORP'
)
agent = AutoGPT.from_llm_and_tools(
    ai_name="Гигачат",
    ai_role="Ассистент",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever(),
)
# Отладка модели
agent.chain.verbose = True

