from transformers import pipeline
import logging
logging.basicConfig(level=logging.INFO)
p = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1, dtype="float32")
p("tests/fixtures/sample_audio.wav", clean_up_tokenization_spaces=False)
