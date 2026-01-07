from fastapi import FastAPI, Request
from fastapi.responses import Response
from langchain_core.messages import HumanMessage
from twilio.twiml.messaging_response import MessagingResponse
from image_download_helper import download_twilio_image_as_base64
from workflow import build_langgraph_app


app = FastAPI()

langgraph_app = build_langgraph_app()

@app.post("/whatsapp/webhook")
async def whatsapp_webhook(request: Request):
    form_data = await request.form()

    user_message = form_data.get("Body", "")
    from_number = form_data.get("From")   # whatsapp:+91xxxx

    # --- Image handling ---
    num_media = int(form_data.get("NumMedia", 0))
    image_url = None

    if num_media > 0:
        media_type = form_data.get("MediaContentType0", "")
        if "image" in media_type:
            raw_url = form_data.get("MediaUrl0")
            image_url = download_twilio_image_as_base64(raw_url)

    # --- LangGraph invocation ---
    config = {
        "configurable": {
            "thread_id": from_number  # per-user memory
        }
    }


    result = langgraph_app.invoke(
        {
            "messages": [HumanMessage(content=user_message)],
            "image_url": image_url
        },
        config=config
    )

    reply = result["messages"][-1].content

    twilio_response = MessagingResponse()
    twilio_response.message(reply)

    return Response(
        content=str(twilio_response),
        media_type="application/xml"
    )
