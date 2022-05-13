from flask import Flask, render_template, request, Response
from TextToVideo import text_to_video
from TextToAudio import text_to_audio
import runner

app = Flask(__name__, static_folder='static/assets',
            template_folder='templates')
# flask_index.html
posts = []


@app.route('/', methods=["GET", "POST"])
@app.route('/services', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        option = request.form.get("file-type")
        text = request.form.get("content")
        if option == 'ttv':
            text_to_video(text)
            key = 0
        if option == 'tta':
            text_to_audio(text)
            key = 1
        if option == 'ttp':
            text = runner.count_fingers()
            key = 2
        posts.append((key, text))
    return render_template("services.html", posts=posts)


'''
# camera.html
camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera')
def view():
    """Video streaming home page."""
    return render_template('camera.html')'''


if __name__ == "__main__":
    app.run(debug=True)
