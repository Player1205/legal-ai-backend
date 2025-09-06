from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],   # for testing only. Replace "*" with your frontend URL for production.
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    # Define a system instruction to guide the model's response style.
    # This is the key change to get a concise answer.
    system_instruction = (
        "You are a concise, helpful, and direct AI assistant. "
        "Answer the user's question as briefly as possible, but with sufficient detail. "
        "Do not include long lists, extensive definitions, or multiple sections."
    )

    # The payload is modified to include the system instruction.
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system_instruction},
                    {"text": request.question}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    try:
        response = requests.post(f"{GEMINI_MODEL_URL}?key={GEMINI_API_KEY}", headers=headers, json=payload)
        
        # Check for non-200 status codes.
        if response.status_code != 200:
            return {"error": f"Error from Gemini: {response.status_code} {response.text}"}

        data = response.json()
        
        # Extract the text response from the correct path.
        if "candidates" in data and len(data["candidates"]) > 0:
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"answer": answer_text}
        else:
            return {"error": "No answer returned from Gemini."}

    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}
