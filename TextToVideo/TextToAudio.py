from gtts import gTTS
import os

cwd = os.getcwd()

def text_to_audio(string):
    language = 'en'
    myobj = gTTS(text=string, lang=language, slow=False)
    name=cwd+'/TextToVideo/static/audio.mp3'
    myobj.save(name)
