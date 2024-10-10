import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram import F
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve Telegram bot token from environment variable
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Verify the token is loaded properly (optional)
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("No TELEGRAM_BOT_TOKEN provided. Check your .env file.")

# Initialize bot with the token
bot = Bot(token=TELEGRAM_BOT_TOKEN)


# Initialize dispatcher with the bot
dp = Dispatcher(bot=bot)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Handler for the /start, /help, and /ars commands
@dp.message(Command(commands=['start', 'help', 'ars']))
async def command_start_handler(message: Message):
    response = f"Hello\n\nPowered by AR Shahin.\nYour command is {message.text}"
    await message.answer(response)

# Handler for the /shahin command
@dp.message(Command(commands=['shahin']))
async def command_shahin_handler(message: Message):
    """
    This handler is triggered when the user sends `/shahin` command.
    """
    response = "Hi Shahin!"
    await message.answer(response)


# Echo 
@dp.message(F.text)
async def echo(message: Message):
    if message.text.startswith('/'):
        return
      
    await message.answer(message.text)


async def main():
    """
    Start the bot and begin polling for new updates (messages).
    """
    await dp.start_polling(bot)

# Entry point of the program
if __name__ == "__main__":
    try:
        # Run the bot with asyncio
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # Gracefully stop the bot on interrupt (Ctrl + C)
        print("Bot stopped.")
