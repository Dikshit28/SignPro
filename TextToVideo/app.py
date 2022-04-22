from flask import Flask, render_template, request
from TextToVideo import text_to_video
from TextToAudio import text_to_audio

app = Flask(__name__,static_folder='static',template_folder='templates')

posts = []


@app.route('/', methods=["GET", "POST"])

@app.route('/index', methods=["GET", "POST"])

def index():
    if request.method == "POST":
        option=request.form.get("file-type")
        text = request.form.get("content")
        if option=='ttv':
            text_to_video(text)
            key=0
        if option == 'tta':
            text_to_audio(text)
            key=1
        posts.append((key,text))
    return render_template("index.html", posts=posts)


if __name__ == "__main__":
    app.run(debug=True)