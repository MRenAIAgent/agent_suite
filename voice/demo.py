from dotenv import load_dotenv
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import play

load_dotenv()

api_key = os.getenv('ELEVENLABS_API_KEY')

client = ElevenLabs(
    # api_key='sk_016a72d3c9e092ddb968b229df9f2efd7e114e07faf0bc70',
    api_key=api_key
)

audio = client.text_to_speech.convert(
    text="The first move is what sets everything in motion.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)

