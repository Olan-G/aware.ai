import torch
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
import gradio as gr

MODEL_NAME = "openai/whisper-small"
BATCH_SIZE = 8

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)


def transcribe(file):
    outputs = pipe(file, batch_size=BATCH_SIZE)
    text = outputs["text"]
    return text


mic_transcribe = gr.Interface(
    fn=transcribe,
    inputs=gr.inputs.Audio(source="microphone", type="filepath", optional=True),
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    title="Speech to Text Recognition",
    allow_flagging="never",
)

mic_transcribe.launch(enable_queue=True)