#!pip install git+https://github.com/suno-ai/bark.git

from bark import SAMPLE_RATE, generate_audio, preload_models
from IPython.display import Audio

preload_models()

text = "I am but a lonely child."

audio_array = generate_audio(text_prompt)
Audio(audio_array, rate=SAMPLE_RATE)