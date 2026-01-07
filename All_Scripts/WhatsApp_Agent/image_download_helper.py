import base64
import requests
import os

def download_twilio_image_as_base64(media_url: str) -> str:
    response = requests.get(
        media_url,
        auth=(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        ),
        allow_redirects=True
    )

    response.raise_for_status()

    encoded = base64.b64encode(response.content).decode("utf-8")

    # Guess mime type (Twilio sends correct headers)
    mime_type = response.headers.get(
        "Content-Type", "image/jpeg"
    )

    return f"data:{mime_type};base64,{encoded}"
