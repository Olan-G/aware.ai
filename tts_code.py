#pip install -Uqq WhisperSpeech

import torch
import torch.nn.functional as F

from IPython.display import Markdown, HTML
from whisperspeech.pipeline import Pipeline

pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

pipe.generate_to_notebook("""
Hallo! Your picture seems to be someone holding a cup of coffee in front of a laptop.
""")