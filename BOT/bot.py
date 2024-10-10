import os
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
import logging

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

app = FastAPI()


try:
    llm = ChatOpenAI(
        model="gpt-4",  
        openai_api_key=os.getenv("OPENAI_API_KEY")  
    )
except Exception as e:
    print(e)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your Telegram Bot token from the .env file
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

# Dictionary to store user states (e.g., waiting for input)
user_states = {}

@app.get("/")
def index(text = "hi"):
    try:
        prompt =  PromptTemplate.from_template("Answer this quesiton  {text}")
        chain = LLMChain(llm=llm, prompt=prompt)
        return {"data": "FAST API","ai" : chain.run(text)}
    except Exception as e:
        print(e)
    

@app.post(f"/webhook/{TELEGRAM_BOT_TOKEN}")
async def telegram_webhook(request: Request):
    data = await request.json()
    print(data)
    
    # Handle callback queries or text messages
    if "message" in data:
        chat_id = data['message']['chat']['id']
        query = data['message']['text']
    
        prompt = PromptTemplate.from_template("You are a helpful assistant. Please answer the following question: {query}")
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.invoke(query)["text"]
        send_message(chat_id,response)

    return {"status": "ok"}


@app.post(f"/webhoook/{TELEGRAM_BOT_TOKEN}")
async def telegram__webhook(request: Request):
    data = await request.json()
    print(data)
    
    # Handle callback queries or text messages
    if "message" in data:
        chat_id = data['message']['chat']['id']
        text = data['message']['text']

        # Check if the user is already in a "waiting for input" state
        if chat_id in user_states:
            if user_states[chat_id] == "awaiting_name":
                # User has responded with their name
                name = text
                response_message = f"Nice to meet you, {name}!"
                send_message(chat_id, response_message)

                # Clear the state for the user
                del user_states[chat_id]
            else:
                send_message(chat_id, "I'm not sure what you're saying. Please use /ask to get started.")

        # Handle commands
        elif text == '/start':
            response_message = "Welcome! You can ask me questions with /ask."
            send_message(chat_id, response_message)
        
        elif text == '/ask':
            response_message = "What is your name?"
            send_message(chat_id, response_message)

            # Set the user's state to expect an answer
            user_states[chat_id] = "awaiting_name"

        elif text == '/help':
            response_message = "Available commands:\n/start - Start the bot\n/ask - Ask for your name\n/help - Show this help message"
            send_message(chat_id, response_message)

        else:
            response_message = "Unknown command. Please use /help to see available commands."
            send_message(chat_id, response_message)

    return {"status": "ok"}

# Helper function to send a message
def send_message(chat_id: int, text: str):
    url = TELEGRAM_API_URL
    payload = {
        'chat_id': chat_id,
        'text': text
    }
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error sending message: {response.text}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
