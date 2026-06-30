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
from numpy.core.numeric import ndarray
from scipy.spatial import distance as dist


class BlinkDetector:
    left_eye: dict[str, int] = {
        "P1": 263,  # Angolo esterno
        "P2": 385,  # Palpebra superiore
        "P3": 387,  # Palpebra superiore
        "P4": 362,  # Angolo interno
        "P5": 373,  # Palpebra inferiore
        "P6": 380,  # Palpebra inferiore
    }
    right_eye: dict[str, int] = {
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
        left_ear_threshold: float = 0.16,
        right_ear_threshold: float = 0.16,
        min_blink_time_threshold: int = 80,
        max_blink_time_threshold: int = 400,
    ) -> None:
        # Soglie di apertura dell'occhio
        # self.EAR_THRESHOLD: float = ear_threshold  # Soglia di apertura dell'occhio
        self.LEFT_EAR_THRESHOLD: float = left_ear_threshold
        self.RIGHT_EAR_THRESHOLD: float = right_ear_threshold
        self.MIN_BLINK_TIME_THRESHOLD: int = min_blink_time_threshold
        self.MAX_BLINK_TIME_THRESHOLD: int = max_blink_time_threshold

        # Tolleranza della differenza accettabile affinché si possa distinguere un occhio chiuso involontariamente
        # per il tiraggio della pelle nel tentativo di chiuderne uno solo
        self.EAR_DIFF: float = 0.05

        self.is_calibrating: bool =

        # Contatori del numero di chiusure degli occhi
        self.left_blink_counter: int = 0
        self.right_blink_counter: int = 0

        # Contatori di tempo per il quale l'occhio è stato chiuso
        self.left_blink_time_counter: int | None = None
        self.right_blink_time_counter: int | None = None

        # Funzione di callback per il calcolo del risultato del modello
        self.on_blink: Callable[[], None] = lambda: None

        # Funzioni di callback per quando vengono sbattuti gli occhi
        self.on_left_blink_callback: Callable[[], None] = lambda: None
        self.on_right_blink_callback: Callable[[], None] = lambda: None

        # Percorso file del model bundle
        self.model_path: str = "models/face_landmarker.task"

        # Salva il timestamp dell'ultimo timestamp in millisecondi
        self.last_timestamp_ms: int = 0

        # Variabili di conteggio e somma per la calibrazione degli occhi
        self.sum_left_ear: float = 0.0
        self.sum_right_ear: float = 0.0
        self.count_ear: int = 0
        self.calib_start_time: int | None = None

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

    def frame_preparation(self, frame, rgb):

        ### Fase di preparazione dati

        # Creazione oggetto mp.Image, che formatta i dati dei pixel in un formato compatibile con i modelli di MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Calcolo del timestamp per ogni frame
        frame_timestamp_ms: int = int(time.time() * 1000)
        if frame_timestamp_ms <= self.last_timestamp_ms:
            frame_timestamp_ms: int = self.last_timestamp_ms + 1
        self.last_timestamp_ms = frame_timestamp_ms

        ### Fase di esecuzione
        self.face_landmarker.detect_async(mp_image, frame_timestamp_ms)

    def mediapipe_callback(
        self,
        result: vision.FaceLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        def get_pixel_coordinates(eye_dict: dict[str, int]) -> dict[str, np.ndarray]:
            # Nuovo dizionario che invece di salvare l'indice del punto facciale, ne salva le coordinate X e Y in pixel sullo schermo
            pixel_eye_dict: dict[str, np.ndarray] = {}

            for key, index in eye_dict.items():
                # Dai dati del volto, viene estratto il punto del volto equivalente all'indice iterato e assegnato a face_point_data.
                # Essendo che ogni punto del volto ha 3 attributi X, e Y adesso face_point_data ha i 3 punti di face_landmarks[index]
                face_point_data = face_landmarks[index]

                # A ogni iterazione viene aggiunto al nuovo dizionario il prodotto tra gli attributi delle coordinate X, Y, Z di face_point_data
                # e la larghezza e altezza del frame.
                # Il prodotto viene racchiuso in un np.array[...] (vettore di NumPy), in quanto la funzione dist.euclidian usata nel
                # metodo ear_math richiede che i punti facciali siano formattati in questa maniera.
                pixel_eye_dict[key] = np.array(
                    [face_point_data.x * width, face_point_data.y * height]
                )

            return pixel_eye_dict

        def precision_filter():
            is_left_eye_closed: bool = sx_ear < self.LEFT_EAR_THRESHOLD
            is_right_eye_closed: bool = dx_ear < self.RIGHT_EAR_THRESHOLD

            # Filtro "Anti-Rumore" per l'occhio Sinistro
            if is_left_eye_closed and (dx_ear - sx_ear) > self.EAR_DIFF:
                if self.left_blink_time_counter is None:
                    self.left_blink_time_counter = timestamp_ms
            elif not is_left_eye_closed:
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
            if is_right_eye_closed and (sx_ear - dx_ear) > self.EAR_DIFF:
                if self.right_blink_time_counter is None:
                    self.right_blink_time_counter = timestamp_ms
            elif not is_right_eye_closed:
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
        sx_ear: float = self.ear_math(
            eye_coordinates=left_eye_coordinates,
        )

        # Calcolo dell'EAR per l'occhio Destro (prospettiva umana)
        dx_ear: float = self.ear_math(
            eye_coordinates=right_eye_coordinates,
        )

        # Chiamata alla funzione
        if self.CALIBRATION:
            self.calibration(sx_ear=sx_ear, dx_ear=dx_ear, timestamp_ms=timestamp_ms)
        else:
            precision_filter()

    def ear_math(self, eye_coordinates) -> float:
        # Aliases
        P1 = eye_coordinates["P1"]
        P2 = eye_coordinates["P2"]
        P3 = eye_coordinates["P3"]
        P4 = eye_coordinates["P4"]
        P5 = eye_coordinates["P5"]
        P6 = eye_coordinates["P6"]

        # Calcolo EAR (Eye Aspect Ratio)
        numerator: float = dist.euclidean(P2, P6) + dist.euclidean(P3, P5)
        denominator: float = 2 * dist.euclidean(P1, P4)
        EAR: float = numerator / denominator

        return EAR

    def calibration(self, sx_ear: float, dx_ear: float, timestamp_ms: int):
        if self.calib_start_time is None:
            self.calib_start_time = timestamp_ms
            print(
                "Inizio calibrazione: Guarda la telecamera con espressione neutra per 3 secondi.\n"
            )

        self.sum_left_ear += sx_ear
        self.sum_right_ear += dx_ear
        self.count_ear += 1

        time_elapsed = timestamp_ms - self.calib_start_time

        if time_elapsed >= 3000:
            AVG_LEFT_EAR: float = self.sum_left_ear / self.count_ear
            AVG_RIGHT_EAR: float = self.sum_right_ear / self.count_ear

            self.LEFT_EAR_THRESHOLD = AVG_LEFT_EAR
            self.RIGHT_EAR_THRESHOLD = AVG_RIGHT_EAR

            print("Calibrazione completata.\n")
            print(
                f"EAR SINISTRO: {self.LEFT_EAR_THRESHOLD} EAR DESTRO: {self.RIGHT_EAR_THRESHOLD}"
            )

            self.CALIBRATION = False
            self.calib_start_time = None
