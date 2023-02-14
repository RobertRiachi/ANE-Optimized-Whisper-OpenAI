import torch
import time
from model import WhisperANE
from whisper import load_model

def load_models(model_name):
    model = load_model(model_name)
    return model, model.dims

def transcribe(whisperANE, audio_path):
    language = "en" if model_name.endswith(".en") else None

    start = time.time()
    result = whisperANE.transcribe(audio_path, language=language, temperature=0.0)
    print(f"took: {time.time() - start}s")

    transcription = result["text"].lower()
    print(transcription)

if __name__ == "__main__":

    # Simple test method to ensure this model still runs end to end
    audio_path = "padthai.mp3"
    model_name = "small"
    whisper, model_dims = load_models(model_name)
    whisper.eval()

    whisperANE = WhisperANE(model_dims).eval()
    whisperANE.load_state_dict(whisper.state_dict())

    transcribe(whisperANE, audio_path)