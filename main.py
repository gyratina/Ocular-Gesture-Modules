###
# project: Ocular Gestures Module (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: blinkDetector.py
###


from pynput.keyboard import Controller, Key

from blinkDetector import BlinkDetector


def on_left_blink():
    print("Sbattuto Occhio SINISTRO\n")
    tastiera.press(Key.cmd_l)
    tastiera.release(Key.cmd_l)


def on_right_blink():
    print("Sbattuto Occhio DESTRO\n")


# Istanziamento di BlinkDetector
blink_detector = BlinkDetector()


def on_calibration(left_eye: float, right_eye: float):
    blink_detector.left_ear_threshold = left_eye
    blink_detector.left_ear_threshold = right_eye
    print("Calibrazione completata.\n")
    print(f"EAR SINISTRO: {left_eye} EAR DESTRO: {right_eye}")


tastiera = Controller()

blink_detector.on_left_blink_callback = on_left_blink
blink_detector.on_right_blink_callback = on_right_blink
blink_detector.on_calibration_callback = on_calibration

blink_detector.start(mode="detect")
