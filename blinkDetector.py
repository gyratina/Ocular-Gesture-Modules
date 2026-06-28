###
# project: Ocular Gestures Module (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: blinkDetector.py
###


import time
from typing import Callable

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

    # Metodo costruttore
    def __init__(
        self,
        ear_threshold=0.16,
        min_blink_time_threshold=80,
        max_blink_time_threshold=400,
    ) -> None:
        self.EAR_THRESHOLD = ear_threshold  # Soglia di apertura dell'occhio
        self.MIN_BLINK_TIME_THRESHOLD = min_blink_time_threshold
        self.MAX_BLINK_TIME_THRESHOLD = max_blink_time_threshold

        # Contatori del numero di chiusure degli occhi
        self.left_blink_counter = 0
        self.right_blink_counter = 0

        # Contatori di tempo per il quale l'occhio è stato chiuso
        self.left_blink_time_counter = None
        self.right_blink_time_counter = None

        # Funzione di callback per il calcolo del risultato del modello
        self.on_blink: Callable[[], None] = None

        # Funzioni di callback per quando vengono sbattuti gli occhi
        self.on_left_blink_callback: Callable[[], None] = None
        self.on_right_blink_callback: Callable[[], None] = None

        # Percorso file del model bundle
        self.model_path = "models/face_landmarker.task"

        # Salva il timestamp dell'ultimo timestamp in millisecondi
        self.last_timestamp_ms: int = 0

        # Impostazioni del modello di Landmarking facciale
        BaseOptions = py.BaseOptions
        FaceLandmarkerOptions = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.mediapipe_callback,
        )
        FaceLandmarker = vision.FaceLandmarker

        self.face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions)

    def close(self):
        self.face_landmarker.close()

    def process_frame(self, frame, rgb):

        ### Fase di preparazione dati

        # Creazione oggetto mp.Image, che formatta i dati dei pixel in un formato compatibile con i modelli di MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Calcolo del timestamp per ogni frame
        frame_timestamp_ms = int(time.time() * 1000)
        if frame_timestamp_ms <= self.last_timestamp_ms:
            frame_timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = frame_timestamp_ms

        ### Fase di esecuzione
        self.face_landmarker.detect_async(mp_image, frame_timestamp_ms)

    def mediapipe_callback(
        self,
        result: vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
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

        # Controllo se la telecamera ha trovato almeno un volto
        if not result.face_landmarks:
            return

        # Ottenimento dati sulla dimensione della camera
        height, width = output_image.height, output_image.width

        # Vengono salvati i dati in merito alla prima faccia trovata da MediaPipe (478 oggetti NormalizedLandmark)
        face_landmarks = result.face_landmarks[0]

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

        # Filtro "Anti-Rumore" per l'occhio Sinistro
        if sx_ear < self.EAR_THRESHOLD and self.left_blink_time_counter is None:
            self.left_blink_time_counter = timestamp_ms
        elif sx_ear >= self.EAR_THRESHOLD:
            if self.left_blink_time_counter is not None:
                blink_time = timestamp_ms - self.left_blink_time_counter

                if (
                    self.MIN_BLINK_TIME_THRESHOLD
                    <= blink_time
                    <= self.MAX_BLINK_TIME_THRESHOLD
                ):
                    self.left_blink_counter += 1
                    # Chiamata a funzione di callback
                    if self.on_left_blink_callback is not None:
                        self.on_left_blink_callback()

            self.left_blink_time_counter = None

        # Filtro "Anti-Rumore" per l'occhio Destro
        if dx_ear < self.EAR_THRESHOLD and self.right_blink_time_counter is None:
            self.right_blink_time_counter = timestamp_ms
        elif dx_ear >= self.EAR_THRESHOLD:
            if self.right_blink_time_counter is not None:
                blink_time = timestamp_ms - self.right_blink_time_counter

                if (
                    self.MIN_BLINK_TIME_THRESHOLD
                    <= blink_time
                    <= self.MAX_BLINK_TIME_THRESHOLD
                ):
                    self.right_blink_counter += 1
                    # Chiamata a funzione di callback
                    if self.on_right_blink_callback is not None:
                        self.on_right_blink_callback()

            self.right_blink_time_counter = None

    def ear_math(self, eye_coordinates) -> float:
        # Aliases
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
