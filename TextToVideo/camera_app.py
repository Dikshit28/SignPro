from pickle import TRUE
from flask import Flask, render_template, Response
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='templates')

# use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
model = keras.models.load_model(r"best_model_dataflair3.h5")

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350


def cal_accum_avg(frame, accumulated_weight):

    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)


def segment_hand(frame, threshold=25):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Fetching contours in the frame (These contours can be of hand or any other object in foreground) ...
    # image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours, hierarchy = cv2.findContours(
        thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If length of contours list = 0, means we didn't get any contours...
    if len(contours) == 0:
        return None
    else:
        # The largest external contour should be the hand
        hand_segment_max_cont = max(contours, key=cv2.contourArea)

        # Returning the hand segment(max contour) and the thresholded image of hand...
        return (thresholded, hand_segment_max_cont)


cam = cv2.VideoCapture(0)


def pred(frame):
    while True:
        num_frames = 0
        word_dict = {0: 'One', 1: 'Ten', 2: 'Two', 3: 'Three', 4: 'Four',
                     5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

        # filpping the frame to prevent inverted image of captured frame...
        frame = cv2.flip(frame, 1)

        frame_copy = frame.copy()

        # ROI from the frame
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 70:

            cal_accum_avg(gray_frame, accumulated_weight)

            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                        (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        else:
            # segmenting the hand region
            hand = segment_hand(gray_frame)

            # Checking if we are able to detect the hand...
            if hand is not None:

                thresholded, hand_segment = hand

                # Drawing contours around hand segment
                cv2.drawContours(
                    frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

                cv2.imshow("Thesholded Hand Image", thresholded)

                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(
                    thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(
                    thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))

                pred = model.predict(thresholded)
                cv2.putText(frame_copy, word_dict[np.argmax(
                    pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw ROI on frame_copy
            cv2.rectangle(frame_copy, (ROI_left, ROI_top),
                          (ROI_right, ROI_bottom), (255, 128, 0), 3)

            # incrementing the number of frames for tracking

            # Display the frame with segmented hand
            cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _",
                        (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
            num_frames += 1
            yield word_dict[np.argmax(pred)]


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = cam.read()  # read the camera frame
        if not success:
            break
        else:
            num_frames = 0
            word_dict = {0: 'One', 1: 'Ten', 2: 'Two', 3: 'Three', 4: 'Four',
                         5: 'Five', 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine'}

            # filpping the frame to prevent inverted image of captured frame...
            frame = cv2.flip(frame, 1)

            frame_copy = frame.copy()

            # ROI from the frame
            roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

            gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

            if num_frames < 70:

                cal_accum_avg(gray_frame, accumulated_weight)

                cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT",
                            (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            else:
                # segmenting the hand region
                hand = segment_hand(gray_frame)

                # Checking if we are able to detect the hand...
                if hand is not None:

                    thresholded, hand_segment = hand

                    # Drawing contours around hand segment
                    cv2.drawContours(
                        frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)

                    cv2.imshow("Thesholded Hand Image", thresholded)

                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(
                        thresholded, cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(
                        thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))

                    pred = model.predict(thresholded)
                    cv2.putText(frame_copy, word_dict[np.argmax(
                        pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Draw ROI on frame_copy
                cv2.rectangle(frame_copy, (ROI_left, ROI_top),
                              (ROI_right, ROI_bottom), (255, 128, 0), 3)

                # incrementing the number of frames for tracking

                # Display the frame with segmented hand
                cv2.putText(frame_copy, "DataFlair hand sign recognition_ _ _",
                            (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield [(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'), num_frames, word_dict[np.argmax(pred)]]  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    ret = gen_frames()
    frames = ret[0]
    word = ret[2]

    return Response(frames, mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('camera.html')


if __name__ == '__main__':
    app.run(debug=True)
