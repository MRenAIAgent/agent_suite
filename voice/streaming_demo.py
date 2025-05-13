from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('ELEVENLABS_API_KEY')

client = ElevenLabs(
    # api_key='sk_016a72d3c9e092ddb968b229df9f2efd7e114e07faf0bc70',
    api_key=api_key
)
audio_stream = client.text_to_speech.convert_as_stream(
    text="This is a test",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2"
)

# option 1: play the streamed audio locally
stream(audio_stream)

# option 2: process the audio bytes manually
for chunk in audio_stream:
    if isinstance(chunk, bytes):
        print(chunk)
