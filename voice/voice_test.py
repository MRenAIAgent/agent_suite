import requests
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play


def test_voice(text: str):
    """
    Send text to voice API and play the returned audio
    
    Args:
        text (str): The text to convert to speech
    """
    try:
        # Make the request to your local API
        response = requests.post(
            'http://localhost:8001/tts',
            json={'text': text}
        )
        
        # Check if request was successful
        response.raise_for_status()
        
        # Play the audio
        audio = AudioSegment.from_mp3(BytesIO(response.content))
        play(audio)
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Error playing audio: {e}")

if __name__ == "__main__":
    # Example usage
    test_text = "Hello, this is a test of the text to speech system."
    test_voice(test_text)