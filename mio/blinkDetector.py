###
# project: Ocular Gestures Module (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: blinkDetector.py
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
    # ? eyes = [left_eye, right_eye]

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

        self.last_timestamp_ms: int = 0

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

        self.face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions)

    def close(self):
        self.face_landmarker.close()

    def process_frame(self, frame, rgb):

        def get_pixel_coordinates(eye_dict: dict[str, int]):
            # Nuovo dizionario che invece di salvare l'indice del punto facciale, ne salva le coordinate X e Y in pixel sullo schermo
            pixel_eye_dict = {}

            for key, index in eye_dict.items():
                # Dai dati del volto, viene estratto il punto del volto equivalente all'indice iterato e assegnato a face_point_data.
                # Essendo che ogni punto del volto ha 3 attributi X, Y, Z, adesso face_point_data ha i 3 punti di face_landmarks[index]
                face_point_data = face_landmarks[index]

                # A ogni iterazione viene aggiunto al nuovo dizionario il prodotto tra gli attributi delle coordinate X e Y di face_point_data
                # e la larghezza e altezza del frame.
                # Il prodotto viene racchiuso in un np.array[...] (vettore di NumPy), in quanto la funzione dist.euclidian usata nel
                # metodo ear_math richiede che i punti facciali siano formattati in questa maniera.
                pixel_eye_dict[key] = np.array(
                    [face_point_data.x * width, face_point_data.y * height]
                )

            return pixel_eye_dict

        # Fase di preparazione dati
        height, width, _ = frame.shape  # Ottenimento dati sulla dimensione della camera

        # Creazione oggetto mp.Image, che formatta i dati dei pixel in un formato compatibile con i modelli di MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Calcolo del timestamp per ogni frame
        frame_timestamp_ms = int(time.time() * 1000)
        if frame_timestamp_ms <= self.last_timestamp_ms:
            frame_timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = frame_timestamp_ms

        # Fase di esecuzione
        face_landmarker_result = self.face_landmarker.detect_for_video(
            mp_image, frame_timestamp_ms
        )

        # Controllo se la telecamera ha trovato almeno un volto
        if not face_landmarker_result.face_landmarks:
            return

        # Vengono salvati i dati in merito alla prima faccia trovata da MediaPipe (478 oggetti NormalizedLandmark)
        face_landmarks = face_landmarker_result.face_landmarks[0]

        # Traduzione dei dizionari dei punti degli occhi in coordinate X, Y dei pixel sullo schermo
        left_eye_coordinates = get_pixel_coordinates(self.left_eye)
        right_eye_coordinates = get_pixel_coordinates(self.right_eye)

        # Calcolo dell'EAR per l'occhio Sinistro (prospettiva umana)
        sx_ear = self.ear_math(
            eye_coordinates=left_eye_coordinates,
        )

        # Calcolo dell'EAR per l'occhio Destro (prospettiva umana)
        dx_ear = self.ear_math(
            eye_coordinates=right_eye_coordinates,
        )

    def ear_math(self, eye_coordinates) -> float:
        # TODO: Scrivere il codice che calcola l'EAR

        # Macro
        P1 = eye_coordinates["P1"]
        P2 = eye_coordinates["P2"]
        P3 = eye_coordinates["P3"]
        P4 = eye_coordinates["P4"]
        P5 = eye_coordinates["P5"]
        P6 = eye_coordinates["P6"]

        # Calcolo EAR (Eye Aspect Ratio)
        numerator = dist.euclidean(P2, P6) + dist.euclidean(P3, P5)
        denominator = 2 * dist.euclidean(P1, P4)
        EAR = numerator / denominator

        return EAR
