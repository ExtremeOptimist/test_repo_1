import functools
from flask import (
    Blueprint, flash, g, redirect, Response, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash
from flaskr.db import get_db
import cv2

bp = Blueprint('/images', __name__)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        # success, frame = camera.read()  # read the camera frame
        frame = cv2.imread('./flaskr/images/bilde_mobil_2.jpg')
        # if not success:
        #    break
        # else:
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@bp.route('/images')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')