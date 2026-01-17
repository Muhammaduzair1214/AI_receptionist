import os
import json
import asyncio
import base64
import re
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
import websockets
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL")
client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"

SYSTEM_INSTRUCTIONS = (
    "You are a friendly AI receptionist. "
    "Greet the user first and ask politely how you can help. "
    "Help users book appointments. "
    "Ask politely for any missing details (name, email, phone, service, date, time). "
    "Once all details are collected, use the book_appointment tool to finalize the booking."
)

chat_history = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]

tools = [
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment after collecting all the necessary details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the person."},
                    "email": {"type": "string", "description": "Email address."},
                    "phone": {"type": "string", "description": "Phone number."},
                    "service": {"type": "string", "description": "The service they want to book."},
                    "date": {"type": "string", "description": "The date of the appointment, e.g., YYYY-MM-DD."},
                    "time": {"type": "string", "description": "The time of the appointment, e.g., HH:MM."},
                },
                "required": ["name", "email", "phone", "service", "date", "time"],
            },
        },
    }
]

class ChatMessage(BaseModel):
    message: str

def send_to_make(data):
    if not MAKE_WEBHOOK_URL:
        print("[WARN] MAKE_WEBHOOK_URL not set. Skipping webhook.")
        return False
    try:
        r = requests.post(MAKE_WEBHOOK_URL, json=data)
        return r.status_code == 200
    except Exception as e:
        print(f"[ERROR] Failed to send to make.com: {e}")
        return False

def process_user_message(user_message: str):
    if user_message:
        chat_history.append({"role": "user", "content": user_message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            function_args = json.loads(tool_calls[0].function.arguments)
            function_args["action"] = "book"
            
            print(f"[INFO] Booking tool called with args: {function_args}")
            success = send_to_make(function_args)
            
            reply = (
                f"✅ Your appointment for {function_args.get('service', 'your service')} "
                f"on {function_args.get('date', 'the selected date')} at {function_args.get('time', 'the selected time')} is booked."
                if success else
                "❌ Booking failed. Please try again."
            )
        else:
            reply = response_message.content

    except Exception as e:
        print(f"[ERROR] OpenAI API call failed: {e}")
        reply = "Sorry, I'm having trouble connecting to my brain right now. Please try again later."

    chat_history.append({"role": "assistant", "content": reply})
    return reply

def check_for_booking(history):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            function_args = json.loads(tool_calls[0].function.arguments)
            function_args["action"] = "book"
            
            print("[INFO] Booking tool called in voice conversation, sending to make.com")
            send_to_make(function_args)

    except Exception as e:
        print(f"[ERROR] Failed to check for booking in voice: {e}")


@app.get("/")
async def home():
    return FileResponse("chat.html")

@app.get("/voice")
async def voice_page():
    return FileResponse("voice.html")

@app.post("/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    reply = process_user_message(body.get("message", ""))
    return JSONResponse({"reply": reply})

@app.websocket("/ws-voice")
async def ws_voice(websocket: WebSocket):
    await websocket.accept()

    session_chat_history = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS.replace(
            "Wrap it in triple backticks with json.", ""
        )}
    ]

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    print("[DEBUG] Connecting to OpenAI Realtime API...")
    print(f"[DEBUG] API Key present: {bool(OPENAI_API_KEY)}")

    try:
        async with websockets.connect(
            OPENAI_REALTIME_URL,
            additional_headers=headers,
            ping_interval=200,
            ping_timeout=20,
            close_timeout=10,
        ) as openai_ws:
            print("[DEBUG] Connected to OpenAI Realtime API successfully!")

            # Enable automatic turn detection
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1", "language": "en"},
                    "turn_detection": {
                        "type": "server_vad"
                    },
                },
            }
            await openai_ws.send(json.dumps(session_config))
            print("[DEBUG] Session config sent")

            current_ai_response = ""

            async def receive_from_client():
                """
                Browser -> Server:
                  - binary frames: raw PCM16 bytes (24k from browser resampler)
                  - text frames: ignored
                """
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg.get("bytes"):
                            data = msg["bytes"]
                            audio_b64 = base64.b64encode(data).decode("utf-8")
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64,
                            }))

                except WebSocketDisconnect:
                    print("[DEBUG] Client disconnected")
                except Exception as e:
                    print(f"[DEBUG] Error receiving from client: {e}")

            async def receive_from_openai():
                nonlocal current_ai_response
                try:
                    async for message in openai_ws:
                        event = json.loads(message)
                        event_type = event.get("type")

                        if event_type == "response.audio.delta":
                            pcm_bytes = base64.b64decode(event.get("delta", ""))
                            await websocket.send_bytes(pcm_bytes)

                        elif event_type == "response.audio_transcript.delta":
                            delta = event.get("delta", "")
                            current_ai_response += delta
                            await websocket.send_json({
                                "type": "transcript",
                                "text": delta,
                            })

                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            user_transcript = event.get("transcript", "")
                            if user_transcript:
                                session_chat_history.append({"role": "user", "content": user_transcript})
                                check_for_booking(session_chat_history)
                            
                            await websocket.send_json({
                                "type": "user_transcript",
                                "text": user_transcript,
                            })

                        elif event_type == "response.done":
                            if current_ai_response:
                                session_chat_history.append({"role": "assistant", "content": current_ai_response})
                                current_ai_response = ""
                            await websocket.send_json({"type": "response_end"})

                        elif event_type == "error":
                            error_msg = event.get("error", {}).get("message", "Unknown error")
                            print(f"[DEBUG] OpenAI error: {error_msg}")
                            try:
                                await websocket.send_json({"type": "error", "message": error_msg})
                            except:
                                pass

                except websockets.exceptions.ConnectionClosed as e:
                    print(f"[DEBUG] OpenAI connection closed: {e.code} {e.reason}")
                except Exception as e:
                    print(f"[DEBUG] OpenAI WebSocket error: {e}")
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "message": "An error occurred with the AI service.",
                        })
                    except:
                        pass

            await asyncio.gather(receive_from_client(), receive_from_openai())

    except websockets.exceptions.InvalidStatusCode as e:
        error_msg = f"OpenAI connection failed with status {e.status_code}"
        print(f"[ERROR] {error_msg}")
        try:
            await websocket.send_json({"type": "error", "message": error_msg})
        except:
            pass
        await websocket.close()

    except Exception as e:
        error_msg = f"Connection error: {type(e).__name__}: {str(e)}"
        print(f"[ERROR] {error_msg}")
        try:
            await websocket.send_json({"type": "error", "message": error_msg})
        except:
            pass
        await websocket.close()