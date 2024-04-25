from config import cred
from langchain_community.chat_models import GigaChat

chat = GigaChat(
    credentials=cred,
    verify_ssl_certs=False,
    scope='GIGACHAT_API_CORP')
print(chat.get_models())
