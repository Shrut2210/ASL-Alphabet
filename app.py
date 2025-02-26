from flask import Flask, render_template, request, jsonify
import os
import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np

model = tf.keras.models.load_model("./30_binary_model.h5")
IMG_SIZE = 32

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.download_utils

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/dashboard", methods=['GET'])
def dashboard():
    return render_template('dashboad.html')

@app.route("/predict", methods=['POST'])
def alphabet_recognize():
    file = request.files['frame']
    nparr = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cap = cv2.VideoCapture(0)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w - 30))
            y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h - 30))
            x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w + 30))
            y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h + 30))

            hand_frame = frame[y_min:y_max, x_min:x_max]

            if hand_frame.size != 0:
                padded_hand_frame = cv2.resize(hand_frame, (32, 32))
                padded_hand_frame = np.array(padded_hand_frame, dtype=np.float32) / 255.0 
                padded_hand_frame = np.expand_dims(padded_hand_frame, axis=0) 

                predictions = model.predict(padded_hand_frame)
                confidence = np.max(predictions)
                label_index = np.argmax(predictions)

                if confidence > 0.7: 
                    label = classes[label_index]
                else:
                    label = "Uncertain"
                    
                print(label)
                
                return jsonify({"label" : label, "confidence": float(confidence)})
        

if __name__ == '__main__':
    app.run(debug=True, port=3000)