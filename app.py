import streamlit as st
import cv2 
import tensorflow as tf
import mediapipe as mp
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

st.title("ASL Alphabet Recognition")

model = tf.keras.models.load_model("./30_binary_model.h5")

IMG_SIZE = 100
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
stream_frame = st.empty()
i = 0

while True:
    ret, frame = cap.read()
    
    if not ret: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w - 30))
            y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h - 30))
            x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w + 30))
            y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h + 30))
            
            cropped_hand = frame[y_min:y_max, x_min:x_max]
            
            if cropped_hand.size != 0 :
                resized_hand = cv2.resize(cropped_hand, (IMG_SIZE, IMG_SIZE))
                landmark_tensor = []
                landmark_tensor = np.zeros((IMG_SIZE, IMG_SIZE))
                
                for idx in range(0,21) :
                    x = int(hand_landmarks.landmark[idx].x * w)
                    y = int(hand_landmarks.landmark[idx].y * h)
                    
                    x_norm = min(int(((x - x_min) / (x_max - x_min)) * (IMG_SIZE - 1)), IMG_SIZE - 1)
                    y_norm = min(int(((y - y_min) / (y_max - y_min)) * (IMG_SIZE - 1)), IMG_SIZE - 1)
                    
                    landmark_tensor[y_norm, x_norm] = 1
                
                landmark_tensor = landmark_tensor.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                
                prediction = model.predict(landmark_tensor)
                confidence = np.max(prediction)
                predicted_class = classes[np.argmax(prediction)]
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stream_frame.image(frame, channels="RGB")
    
if st.button("Stop Camera", key=1):
    cap.release()
    cv2.destroyAllWindows()