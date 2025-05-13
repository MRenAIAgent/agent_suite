from elevenlabs import stream, play
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv
import os


class ElevenLabsClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        self.client = ElevenLabs(api_key=self.api_key)
        self.default_voice_id = "JBFqnCBsd6RMkjVDRZzb"
        self.default_model = "eleven_multilingual_v2"

    def create_voice(self, name, description=None, labels=None, files=None):
        """
        Create a new voice and return the voice ID
        Args:
            name (str): Name of the voice
            description (str, optional): Description of the voice
            labels (dict, optional): Labels to add to the voice
            files (list, optional): List of audio file paths to use for the voice
        Returns:
            str: The ID of the created voice
        """
        voice = self.client.voices.add(
            name=name,
            description=description,
            labels=labels,
            files=files
        )
        return voice.voice_id

    def tts(self, text, voice_id=None, model_id=None, output_format="mp3_44100_128"):
        """
        Convert text to speech and play audio
        """
        voice_id = voice_id or self.default_voice_id
        model_id = model_id or self.default_model
        
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format
        )
        play(audio)
        return audio

    def tts_streaming(self, text, voice_id=None, model_id=None):
        """
        Convert text to speech and stream audio
        """
        voice_id = voice_id or self.default_voice_id
        model_id = model_id or self.default_model
        
        audio_stream = self.client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=voice_id,
            model_id=model_id
        )
        stream(audio_stream)
        return audio_stream
