from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from starlette.responses import HTMLResponse, JSONResponse
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
BASE_PATH = Path(__file__).resolve().parent

# Configure Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

app.mount("/static", StaticFiles(directory=BASE_PATH / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_PATH / "templates"))

def convert_to_gemini_history(messages):
    """Convert client messages to Gemini-compatible history format"""
    history = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        history.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })
    return history

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask")
async def ask_question(request: Request):
    try:
        data = await request.json()
        messages = data.get("messages", [])
        
        print(f"Received messages: {messages}")  # Debug logging
        
        # Create chat session
        chat = model.start_chat(history=convert_to_gemini_history(messages))
        
        last_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"), 
            ""
        )
        
        print(f"Processing message: {last_user_message}")  
        
        # Generate response
        # when user wants to place an order:
        if "order" in last_user_message.lower():
            response = chat.send_message(
                system_message="""
        You are a senior sales rep at a top hotel in Bhubaneswar. 
        Always list 3–5 genuine, popular Bhubaneswar dishes. 
        Number each line. Start with 🍲. 
        Keep each description ≤ 50 characters. 
        Do NOT invent dishes—use only authentic names.
        """,
                user_message="Please suggest something to order.",
                temperature=0,
                max_tokens=150
            )

        else:
            response = chat.send_message(last_user_message)
        
        return JSONResponse({
            "response": response.text,
            "messages": messages + [{"role": "model", "content": response.text}]
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # This will show in terminal
        import traceback
        traceback.print_exc()  # Print full traceback
        raise HTTPException(status_code=500, detail=str(e))