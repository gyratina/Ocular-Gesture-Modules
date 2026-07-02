###
# project: Ocular Gesture Modules (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: blinkDetector.py
###

import logging
import os
import time
from enum import Enum
from typing import Callable

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist

from camera_config import CameraConfig

log: logging.Logger = logging.getLogger(f"OGM.{__name__}")


class ActionType(Enum):
    """
    Enumeration representing the different types of detected ocular gestures.
    """
    LEFT = 0
    RIGHT = 1
    BOTH = 2


class BlinkDetector:
    """
    Core detector class that processes frames to identify and record voluntary ocular gestures (blinks).
    Supports single eye and both eyes gestures, featuring a state-machine based precision filter.
    """
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
        max_blink_time_threshold: int = 500,
        ear_diff: float = 0.05,
        model_path: str | None = None,
    ) -> None:
        """
        Initializes the BlinkDetector with specific thresholds for gesture detection.

        Args:
            left_ear_threshold (float): EAR threshold to consider the left eye closed.
            right_ear_threshold (float): EAR threshold to consider the right eye closed.
            min_blink_time_threshold (int): Minimum duration (ms) for a closure to be considered a voluntary blink.
            max_blink_time_threshold (int): Maximum duration (ms) for a closure to be considered a voluntary blink.
            ear_diff (float): Tolerance for EAR difference to avoid false asymmetrical blink triggers (e.g. skin pulling).
            model_path (str | None): Absolute path to the MediaPipe Face Landmarker model. If None, uses the bundled model.
        """
        # Soglie di apertura dell'occhio
        # self.EAR_THRESHOLD: float = ear_threshold  # Soglia di apertura dell'occhio
        self.left_ear_threshold: float = left_ear_threshold
        self.right_ear_threshold: float = right_ear_threshold

        # Soglia minima e massima per considerare il battito volontario
        self.min_blink_time_threshold: int = min_blink_time_threshold
        self.max_blink_time_threshold: int = max_blink_time_threshold

        self.actions: list[tuple[ActionType, int | None]] = []
        self.last_reopening_timestamp: int | None = None

        # Tolleranza della differenza di tipo EAR accettabile affinché si possa distinguere un occhio chiuso involontariamente
        # per il tiraggio della pelle nel tentativo di chiuderne uno solo
        self.ear_diff: float = ear_diff

        self.is_calibrating: bool = False
        self.on_calibration_callback: Callable[[float, float], None] | None = None

        self.last_action: ActionType | None = None
        # Contatori di tempo per il quale l'occhio è stato chiuso
        self.blink_time_counter: int | None = None
        self.left_blink_time_counter: int | None = None
        self.right_blink_time_counter: int | None = None
        self.both_blink_time_counter: int | None = None

        # Funzioni di callback per quando viene effettuata una gesture con il battito delle ciglia
        self.on_blink: Callable[[list[tuple[ActionType, int | None]]], None] | None = (
            None
        )

        # Percorso file del model bundle
        if model_path is None:
            current_directory: str = os.path.dirname(__file__)
            self.model_path: str = os.path.join(
                current_directory, "models", "face_landmarker.task"
            )
        else:
            self.model_path = model_path

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

    def close(self) -> None:
        """
        Closes the underlying MediaPipe face landmarker instance and releases its resources.
        """
        self.face_landmarker.close()

    def reset_log(self) -> None:
        """
        Clears the logged actions and resets the combo timer. Usually called after a combo is successfully matched.
        """
        self.actions.clear()
        self.last_reopening_timestamp = None

    def frame_preparation(self, frame, rgb) -> None:

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
    ) -> None:
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

        def precision_filter() -> None:
            is_left_eye_closed: bool = sx_ear < self.left_ear_threshold
            is_right_eye_closed: bool = dx_ear < self.right_ear_threshold
            are_both_eyes_closed: bool = is_left_eye_closed and is_right_eye_closed
            left_eye_filter: bool = (
                is_left_eye_closed and (dx_ear - sx_ear) > self.ear_diff
            )
            right_eye_filter: bool = (
                is_right_eye_closed and (sx_ear - dx_ear) > self.ear_diff
            )

            reopening_moment: int | None = None
            lapse: int | None = None

            current_action: ActionType | None = None
            if are_both_eyes_closed:
                current_action = ActionType.BOTH
            elif left_eye_filter:
                current_action = ActionType.LEFT
            elif right_eye_filter:
                current_action = ActionType.RIGHT

            if current_action is not None and self.blink_time_counter is None:
                self.blink_time_counter = timestamp_ms

            if current_action != self.last_action:
                if self.blink_time_counter is not None and self.last_action is not None:
                    reopening_moment = timestamp_ms
                    blink_time: int = reopening_moment - self.blink_time_counter

                    if (
                        self.min_blink_time_threshold
                        <= blink_time
                        <= self.max_blink_time_threshold
                    ):
                        if not self.actions and self.last_reopening_timestamp is None:
                            self.actions.append((self.last_action, None))
                            self.last_reopening_timestamp = reopening_moment

                        elif self.last_reopening_timestamp is not None:
                            lapse = (
                                self.blink_time_counter - self.last_reopening_timestamp
                            )
                            self.actions[-1] = (self.actions[-1][0], lapse)
                            self.actions.append((self.last_action, None))
                            self.last_reopening_timestamp = reopening_moment

                        # Chiamata a funzione di callback
                        if self.on_blink is not None:
                            self.on_blink(self.actions)

                self.last_action = current_action
                self.blink_time_counter = None

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

        # Se la calibrazione è impostata su True allora avvia la calibrazione,
        # altrimenti continua filtrando le gesture e chiamando funzioni di callback
        if self.is_calibrating:
            self.calibration(sx_ear=sx_ear, dx_ear=dx_ear, timestamp_ms=timestamp_ms)
        else:
            precision_filter()

    def ear_math(self, eye_coordinates) -> float:
        """
        Calculates the Eye Aspect Ratio (EAR) based on the 6 facial landmarks defining an eye.

        Args:
            eye_coordinates (dict): A dictionary mapping 'P1' through 'P6' to numpy coordinate arrays.

        Returns:
            float: The computed EAR value for the given eye.
        """
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

    def calibration(self, sx_ear: float, dx_ear: float, timestamp_ms: int) -> None:
        if self.calib_start_time is None:
            self.calib_start_time = timestamp_ms
            log.info(
                "Inizio calibrazione: Guarda la telecamera con espressione neutra per 3 secondi.\n"
            )

        self.sum_left_ear += sx_ear
        self.sum_right_ear += dx_ear
        self.count_ear += 1

        time_elapsed = timestamp_ms - self.calib_start_time

        if time_elapsed >= 3000:
            AVG_LEFT_EAR: float = (self.sum_left_ear / self.count_ear) * 0.75
            AVG_RIGHT_EAR: float = (self.sum_right_ear / self.count_ear) * 0.75

            self.is_calibrating = False
            self.calib_start_time = None

            if self.on_calibration_callback is not None:
                self.on_calibration_callback(AVG_LEFT_EAR, AVG_RIGHT_EAR)

    def start(
        self, mode: str = "detect", camera_config: CameraConfig | None = None
    ) -> None:
        """
        Starts the internal camera loop and processes frames synchronously.

        Args:
            mode (str): Operational mode. Use "calibrate" for threshold calibration or "detect" for gesture recognition.
            camera_config (CameraConfig | None): Custom camera configuration. If None, default 720p 30fps config is used.
        """
        match mode:
            case "calibrate":
                self.is_calibrating = True
                self.calib_start_time = None
                log.info("Avvio telecamera in modalità CALIBRAZIONE.")
            case _:
                self.is_calibrating = False
                log.info("Avvio telecamera in modalità RILEVAMENTO.")

        if camera_config is None:
            camera_config = CameraConfig()

        video: cv.VideoCapture = camera_config.set_camera()

        while True:
            status, frame = video.read()

            if not status:
                log.error("Errore, impossibile trovare un fotogramma.")
                break

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            self.frame_preparation(frame=frame, rgb=rgb_frame)

            if mode == "calibrate" and self.is_calibrating is False:
                break

            # Display the resulting frame
            cv.imshow("frame", frame)
            if cv.waitKey(1) == ord("q"):
                break

        # When everything done, release the capture
        self.close()
        video.release()
        cv.destroyAllWindows()
