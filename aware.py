# John Olan Gomez
# Yna Gabrielle Foronda
# Clyde Justine Nate
# Krystal Denise Taclas
# June 7, 2024
# Summer School in Industrial AI
# VIVES University of Applied Sciences
# Brugge, Belgium

# Image to text libraries
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Text to speech libraries
from IPython.display import Markdown, HTML
from whisperspeech.pipeline import Pipeline
from playsound import playsound as play

# initialize image-to-text model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# initialize text-to-speech converter
pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model')

# image-to-text function
def img2txt(image):
    # unconditional image captioning
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    description = processor.decode(out[0], skip_special_tokens=True)

    return description

def tts(text):
    filename = 'sound.mp3'
    pipe.generate_to_file(filename, text)
    play(filename, True)


if __name__ == "__main__":
    # input image
    image = Image.open('sample1.jpg')
    # resize image
    res_image = image.resize((224,224))
    # save the resized image
    res_image.save('res_image.jpg')
    # open image as an Image object
    image = Image.open('res_image.jpg')

    # generate image to text
    text = str(img2txt(image))

    # generate audio
    tts(text)
