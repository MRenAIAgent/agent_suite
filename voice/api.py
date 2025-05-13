import asyncio
# In voice/api.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from voice.voice_client import ElevenLabsClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

voice_client = ElevenLabsClient()

class TextRequest(BaseModel):
    text: str

@app.post("/tts")
async def text_to_speech(request: TextRequest):
    audio_generator = voice_client.tts_streaming(request.text)
    
    return StreamingResponse(
        audio_generator,
        media_type="audio/mpeg"
    )


@app.websocket("/stream-voice")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming voice audio from frontend
    
    Accepts streaming audio chunks and processes them in real-time
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive audio chunk from client
            audio_chunk = await websocket.receive_bytes()
            
            # Here you can process the audio chunk in real-time
            # For example, transcribe, analyze, or forward to another service
            
            # Send acknowledgment back to client
            await websocket.send_json({
                "status": "success",
                "bytes_received": len(audio_chunk)
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        await websocket.send_json({
            "status": "error",
            "message": str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)
