from turtle import pos
from flask import Flask, render_template, request
from TextToVideo import text_to_video

app = Flask(__name__,static_folder='static',template_folder='templates')

posts = []


@app.route('/', methods=["GET", "POST"])

@app.route('/index', methods=["GET", "POST"])

def index():
    if request.method == "POST":
        text = request.form.get("content")
        print(text)
        text_to_video(text)
        posts.append(text.upper())
        print(posts,posts[0])
    return render_template("index.html", posts=posts)

if __name__ == "__main__":
    app.run(debug=True)