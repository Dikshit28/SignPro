import ffmpeg
import os

cwd=os.getcwd()

def convert_video_ffmpeg():
    stream=ffmpeg.input(cwd+'/TextToVideo/static/POST.avi')
    stream=ffmpeg.output(stream,cwd+'/TextToVideo/static/POST.webm')
    ffmpeg.run(stream,overwrite_output=True)

convert_video_ffmpeg()