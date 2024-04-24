def standard_answers(input):
    message = str(input).lower()

    if message in ('hello', 'hi', 'sup', 'привет', 'здравствуй'):
        return "Привет! "

    if message in ('who are you', 'who are you?', 'кто ты', 'кто ты?'):
        return "Я Эксперт по вопросам питания ChatGPT, помогаю с определением безопасности продуктов по их составу и даю рекомендации по питанию"

    return ""
