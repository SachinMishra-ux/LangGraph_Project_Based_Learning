from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

load_dotenv(override=True)

app = FastAPI()

llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.1,
    max_tokens=2000
)

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()

    user_message = form_data.get("Body")
    from_number = form_data.get("From")

    print("Incoming WhatsApp message:", user_message)

    if not user_message:
        reply = "I didn’t receive your message. Please try again."
    else:
        response = llm.invoke(user_message)
        reply = response.content

    twilio_response = MessagingResponse()
    twilio_response.message(reply)

    return Response(
        content=str(twilio_response),
        media_type="application/xml"
    )

