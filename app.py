import streamlit as st
import cv2
import tensorflow as tf
import mediapipe as mp
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

model = tf.keras.models.load_model("30_binary_model.h5")

IMG_SIZE = 100
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

st.title("ASL Alphabet Recognition")

class ASLTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = img.shape
                x_min = max(0, int(min([lm.x for lm in hand_landmarks.landmark]) * w - 30))
                y_min = max(0, int(min([lm.y for lm in hand_landmarks.landmark]) * h - 30))
                x_max = min(w, int(max([lm.x for lm in hand_landmarks.landmark]) * w + 30))
                y_max = min(h, int(max([lm.y for lm in hand_landmarks.landmark]) * h + 30))

                cropped_hand = img[y_min:y_max, x_min:x_max]

                if cropped_hand.size != 0:
                    resized_hand = cv2.resize(cropped_hand, (IMG_SIZE, IMG_SIZE))
                    landmark_tensor = np.zeros((IMG_SIZE, IMG_SIZE))

                    for idx in range(21):
                        x = int(hand_landmarks.landmark[idx].x * w)
                        y = int(hand_landmarks.landmark[idx].y * h)
                        x_norm = min(int(((x - x_min) / (x_max - x_min)) * (IMG_SIZE - 1)), IMG_SIZE - 1)
                        y_norm = min(int(((y - y_min) / (y_max - y_min)) * (IMG_SIZE - 1)), IMG_SIZE - 1)
                        landmark_tensor[y_norm, x_norm] = 1

                    landmark_tensor = landmark_tensor.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                    prediction = model.predict(landmark_tensor)
                    confidence = np.max(prediction)
                    predicted_class = classes[np.argmax(prediction)]

                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(img, f"{predicted_class} ({confidence:.2f})", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="asl_recognition", video_transformer_factory=ASLTransformer)

st.write("Show an ASL alphabet sign to the camera, and the model will recognize it in real-time!")
