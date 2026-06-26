import importlib
import os
import time
import urllib.request

import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

# Proviamo ad importare l'API legacy solutions in modo dinamico per evitare segnalazioni di errore da Zed/Pyright
try:
    face_mesh = importlib.import_module("mediapipe.solutions.face_mesh")
    USE_TASKS_API = False
except ModuleNotFoundError:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    USE_TASKS_API = True


class BlinkDetector:
    # Indici dei landmark per l'occhio destro e sinistro in MediaPipe Face Mesh
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_threshold=0.25, k_frame_threshold=6) -> None:
        self.EAR_THRESHOLD = ear_threshold  # Soglia di apertura dell'occhio
        self.K_FRAME_THRESHOLD = k_frame_threshold  # Soglia del numero di fotogrammi necessari per considerare l'occhio chiuso

        self.blink_counter = 0  # Contatore del numero di chiusura degli occhi
        self.frameOfBlink_counter = (
            0  # Contatore di frame nel quale l'occhio è stato chiuso
        )

        self.on_blink_callback = (
            None  # Funzione di callback per quando vengono sbattuti gli occhi
        )

        self.use_tasks_api = USE_TASKS_API

        if self.use_tasks_api:
            # Nuova Tasks API (necessaria per Python 3.13)
            self.model_path = "face_landmarker.task"
            self._ensure_model_exists()

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options, running_mode=vision.RunningMode.VIDEO
            )
            self.landmarker = vision.FaceLandmarker.create_from_options(options)
        else:
            # API Legacy (funziona fino a Python 3.12)
            self.face_mesh = face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True
            )

    def _ensure_model_exists(self):
        """Scarica il modello face_landmarker.task se non è presente localmente."""
        if not os.path.exists(self.model_path):
            print(f"[OGM] Scaricamento del modello '{self.model_path}' in corso...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            try:
                urllib.request.urlretrieve(url, self.model_path)
                print(
                    f"[OGM] Modello scaricato con successo e salvato come '{self.model_path}'."
                )
            except Exception as e:
                print(f"[OGM] Errore durante il download del modello: {e}")
                raise

    def calculate_ear(self, landmarks, eye_indices, width, height):
        # Estrae i 6 punti per l'occhio specificato
        points = []
        for idx in eye_indices:
            lm = landmarks[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)
            points.append((x, y))

        # Calcolo delle distanze verticali e orizzontale
        d_v1 = dist.euclidean(points[1], points[5])  # d(P2, P6)
        d_v2 = dist.euclidean(points[2], points[4])  # d(P3, P5)
        d_h = dist.euclidean(points[0], points[3])  # d(P1, P4)

        if d_h == 0:
            return 0.0

        ear = (d_v1 + d_v2) / (2.0 * d_h)
        return ear

    def process_frame(self, frame):
        """
        Elabora un fotogramma video (in formato BGR da OpenCV),
        rileva i punti del viso, calcola l'EAR medio per entrambi gli occhi
        e aggiorna lo stato dei battiti.
        """
        if frame is None:
            return None, self.blink_counter

        height, width, _ = frame.shape
        # Conversione da BGR (OpenCV) a RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        landmarks = None

        if self.use_tasks_api:
            # Utilizza la nuova Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            # Calcolo timestamp incrementale in millisecondi
            timestamp_ms = int(time.time() * 1000)
            detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            if detection_result.face_landmarks:
                landmarks = detection_result.face_landmarks[0]
        else:
            # Utilizza l'API Legacy
            results = self.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

        ear = None

        if landmarks is not None:
            # Calcola l'EAR per entrambi gli occhi
            ear_right = self.calculate_ear(
                landmarks, self.RIGHT_EYE_INDICES, width, height
            )
            ear_left = self.calculate_ear(
                landmarks, self.LEFT_EYE_INDICES, width, height
            )

            # Media dell'EAR dei due occhi
            ear = (ear_right + ear_left) / 2.0

            # Logica temporale (Filtro anti-rumore)
            if ear < self.EAR_THRESHOLD:
                self.frameOfBlink_counter += 1
            else:
                if self.frameOfBlink_counter >= self.K_FRAME_THRESHOLD:
                    self.blink_counter += 1
                    if self.on_blink_callback:
                        self.on_blink_callback(self.blink_counter)
                self.frameOfBlink_counter = 0
        else:
            # Se il volto non viene rilevato, azzera il contatore di frame
            self.frameOfBlink_counter = 0

        return ear, self.blink_counter

    def close(self):
        """Rilascia le risorse allocate."""
        if hasattr(self, "landmarker") and self.use_tasks_api:
            try:
                self.landmarker.close()
            except Exception:
                pass
        elif hasattr(self, "face_mesh"):
            try:
                self.face_mesh.close()
            except Exception:
                pass

    def __del__(self):
        """Assicura il rilascio delle risorse alla distruzione dell'oggetto."""
        try:
            self.close()
        except:
            pass
