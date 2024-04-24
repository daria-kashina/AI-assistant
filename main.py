import os
import answers
from dotenv import load_dotenv
from telegram import InputTextMessageContent, BotCommand
from telegram.ext import Application, ApplicationBuilder, MessageHandler, CommandHandler, filters

from gigachat_tools.agent import simple_agent


class ScientificAssistantBot:
    def __init__(self):
        self.application = ApplicationBuilder() \
            .token(os.environ['TELEGRAM_BOT_TOKEN']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()

        self.commands = [
            BotCommand(command='help', description="Помощь, описание."),
            BotCommand(command='start', description="Команда старт"),
        ]
        self.gigachain_agent = simple_agent

    @staticmethod
    async def start_command(update, context):
        await update.message.reply_text(f"Привет {update.message.from_user.first_name}! " + \
                                        "Я AI-ассистент.")

    @staticmethod
    async def help_command(update, context):
        await update.message.reply_text("Привет задавай вопрос!")


    async def handle_message(self, update, context):
        text = str(update.message.text).lower()
        # Проверяем, если есть ответ в заранее определенных ответах
        response = answers.standard_answers(text)

        # Если ответ не найден, используем модель GigaChat для получения ответа
        if not response:
            response = await self.gigachain_agent(credentials=os.environ['GIGACHAT_CRED'], scope='GIGACHAT_API_CORP')

        # Отправляем ответ пользователю
        await update.message.reply_text(response)


    @staticmethod
    def error(update, context):
        print(f'Update {update} caused error: {context.error}')


    async def post_init(self, application: Application) -> None:

        # Устанавливаем команды бота 
        await application.bot.set_my_commands(self.commands)

    def start_bot(self):
        self.application.add_handler(CommandHandler('start', self.start_command))
        self.application.add_handler(CommandHandler('help', self.help_command))

        # static_handle_message = lambda update, context: self.handle_message(update, context)
        self.application.add_handler(MessageHandler(filters.TEXT, self.handle_message))

        self.application.add_error_handler(self.error)

        self.application.run_polling()


if __name__ == "__main__":
    # Read .env file
    load_dotenv()

    bot = ScientificAssistantBot()
    print('Bot started')

    bot.start_bot()
    print('Bot stopped')
