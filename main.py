###
# project: Ocular Gesture Modules (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: blinkDetector.py
###


import logging

from pynput.keyboard import Controller, Key

from blink_detector import ActionType, BlinkDetector
from camera_config import CameraConfig

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)

tastiera = Controller()

# Istanziamento di BlinkDetector
blink_detector = BlinkDetector()


def elabora_combinazioni(azioni: list[tuple[ActionType, int | None]]):
    match azioni:
        # Combo: Destro -> Sinistro (con pausa massima di 800 ms)
        case [*_, (ActionType.RIGHT, p_dx), (ActionType.LEFT, _)] if (
            p_dx is not None and p_dx <= 800
        ):
            print(
                "\n🌟 COMBO VELOCE RILEVATA: Destro -> Sinistro! (Apro il terminale...)"
            )
            tastiera.press(Key.cmd_l)
            tastiera.release(Key.cmd_l)
            blink_detector.reset_log()

        # Combo: Sinistro -> Destro (con pausa massima di 800 ms)
        case [*_, (ActionType.LEFT, p_sx), (ActionType.RIGHT, _)] if (
            p_sx is not None and p_sx <= 1000
        ):
            print("\n🌟 COMBO RILEVATA: Sinistro -> Destro! (Eseguo macro...)")
            blink_detector.reset_log()

        # Combo a 3 mosse: Destro -> Sinistro -> Destro (pause massime di 800 ms ciascuna)
        case [
            *_,
            (ActionType.RIGHT, p1),
            (ActionType.LEFT, p2),
            (ActionType.RIGHT, _),
        ] if (p1 is not None and p1 <= 1000) and (p2 is not None and p2 <= 1000):
            print("\n🌟 SUPER COMBO A 3 MOSSE RILEVATA: Destro -> Sinistro -> Destro!")
            blink_detector.reset_log()

        # Combo: Doppio Destro (con pausa massima di 800 ms)
        case [*_, (ActionType.RIGHT, p), (ActionType.RIGHT, _)] if (
            p is not None and p <= 1000
        ):
            print("\n🌟 COMBO RILEVATA: Doppio Destro!")
            blink_detector.reset_log()

        # Qualsiasi altra sequenza o tempo troppo lungo verrà ignorata finché non c'è un match
        case _:
            pass


def on_calibration(left_eye: float, right_eye: float):
    blink_detector.left_ear_threshold = left_eye
    blink_detector.right_ear_threshold = right_eye
    print("Calibrazione completata.\n")
    print(f"EAR SINISTRO: {left_eye} EAR DESTRO: {right_eye}")


blink_detector.on_blink = elabora_combinazioni
blink_detector.on_calibration_callback = on_calibration

my_camera = CameraConfig(camera_index=1)
blink_detector.start(mode="detect", camera_config=my_camera)
