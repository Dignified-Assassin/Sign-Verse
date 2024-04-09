from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)

# Initialize hand detection and classification modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("Model\\keras_model.h5", "Model\\labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H"]

@app.route('/')
def index():
    return render_template('Website/homepage.html')

@app.route('/camera')
def camera():
    return render_template('Website/index.html')

def gen_frames():
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            imgWhite = np.ones((300, 600, 3), np.uint8) * 255
            for hand in hands:
                if hand['type'] == 'Left':
                    xl, yl, wl, hl = hand['bbox']
                    imgCropl = img[yl - offset: yl + hl + offset, xl - offset: xl + wl + offset]
                    imgCropshape = imgCropl.shape

                    ar = hl / wl  # aspect ratio
                    if ar > 1:
                        k = imgSize / hl
                        wcal = math.ceil(wl * k)
                        imgResize = cv2.resize(imgCropl, (wcal, imgSize))
                        wgap = math.ceil((imgSize - wcal) / 2)
                        imgWhite[:300, 300 + wgap:300 + wgap + wcal] = imgResize
                    else:
                        k = imgSize / wl
                        hcal = math.ceil(hl * k)
                        imgResize = cv2.resize(imgCropl, (imgSize, hcal))
                        hgap = math.ceil((imgSize - hcal) / 2)
                        imgWhite[hgap:hgap + hcal, 300:600] = imgResize

                if hand['type'] == 'Right':
                    xr, yr, wr, hr = hand['bbox']
                    imgCropr = img[yr - offset: yr + hr + offset, xr - offset: xr + wr + offset]
                    imgCropshape = imgCropr.shape

                    ar = hr / wr  # aspect ratio
                    if ar > 1:
                        k = imgSize / hr
                        wcal = math.ceil(wr * k)
                        imgResize = cv2.resize(imgCropr, (wcal, imgSize))
                        wgap = math.ceil((imgSize - wcal) / 2)
                        imgWhite[:, wgap:wgap + wcal] = imgResize
                    else:
                        k = imgSize / wr
                        hcal = math.ceil(hr * k)
                        imgResize = cv2.resize(imgCropr, (imgSize, hcal))
                        hgap = math.ceil((imgSize - hcal) / 2)
                        imgWhite[hgap:hgap + hcal, :300] = imgResize

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                if index is not None:
                    emit('prediction', labels[index])  # Send prediction to client

            ret, frame = cv2.imencode('.jpg', imgWhite)
            frame_bytes = frame.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app)
