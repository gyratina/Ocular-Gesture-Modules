###
# project: Ocular Gestures Module (OGM)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# start: 2026-06-26 (yyyy-mm-dd)
###


import time

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist


class BlinkDetector:
    left_eye = {
        "P1": 263,  # Angolo esterno
        "P2": 385,  # Palpebra superiore
        "P3": 387,  # Palpebra superiore
        "P4": 362,  # Angolo interno
        "P5": 373,  # Palpebra inferiore
        "P6": 380,  # Palpebra inferiore
    }
    right_eye = {
        "P1": 33,  # Angolo esterno
        "P2": 160,  # Palpebra superiore
        "P3": 158,  # Palpebra superiore
        "P4": 133,  # Angolo interno
        "P5": 153,  # Palpebra inferiore
        "P6": 144,  # Palpebra inferiore
    }
    eyes = [left_eye, right_eye]

    # Metodo costruttore
    def __init__(self, ear_threshold=0.25, k_frame_threshold=6, running_mode=1) -> None:
        self.EAR_THRESHOLD = ear_threshold  # Soglia di apertura dell'occhio
        self.K_FRAME_THRESHOLD = k_frame_threshold  # Soglia del numero di fotogrammi necessari per considerare l'occhio chiuso

        self.blink_counter = 0  # Contatore del numero di chiusura degli occhi
        self.frameOfBlink_counter = (
            0  # Contatore di frame nel quale l'occhio è stato chiuso
        )

        self.on_blink_callback = (
            None  # Funzione di callback per quando vengono sbattuti gli occhi
        )

        self.model_path = (
            "model_bundle/face_landmarker.task"  # Percorso file del model bundle
        )

        # match che imposta la running mode
        match running_mode:
            case 0:
                self.running_mode = vision.RunningMode.IMAGE

            case 1:
                self.running_mode = vision.RunningMode.VIDEO

            case 2:
                self.running_mode = vision.RunningMode.LIVE_STREAM

        # Impostazioni del modello di Landmarking facciale
        BaseOptions = py.BaseOptions
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=self.running_mode,
            num_faces=1,
        )
        FaceLandmarker = vision.FaceLandmarker

        self.face_land_marker = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions
        )

    def close(self):
        self.face_land_marker.close()

    def process_frame(self, frame):

        # Fase di preparazione dati
        height, width, _ = frame.shape

        # ? Approfondisci
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        

        # Fase di esecuzione
        face_landmarker_result = landmarker.detect_for_video(
            mp_image, frame_timestamp_ms
        )

        sx_ear = self.ear_math(
            landmarks=self.face_land_marker,
            eye_coordinates=None,
            height=height,
            width=width,
        )
        sx_ear = self.ear_math(
            landmarks=self.face_land_marker,
            eye_coordinates=None,
            height=height,
            width=width,
        )

    def ear_math(self, landmarks, eye_coordinates, height, width) -> float:
        pass  # TODO: Scrivere il codice che calcola l'EAR
