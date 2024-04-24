from langchain_community.embeddings.gigachat import GigaChatEmbeddings


async def simple_agent(credentials, scope):
    embeddings = GigaChatEmbeddings(
        credentials=credentials,
        scope=scope,
        verify_ssl_certs=False)
    result = embeddings.embed_documents(texts=["Привет!"])
    # print(result)
    return "Проверка, все хорошо gigachat работает!"
