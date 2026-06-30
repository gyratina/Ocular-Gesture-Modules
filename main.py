###
# project: Ocular Gestures Module (OGM)
# project-start: 2026-06-26 (yyyy-mm-dd)
# author-username: @gyratina on GitHub
# author-name: Valerio Di Tommaso
# author-email: contact.me@valerioditommaso.dev
# file-name: blinkDetector.py
###


import cv2 as cv
from pynput.keyboard import Controller, Key

from blinkDetector import BlinkDetector

video = cv.VideoCapture(1)

if not video.isOpened():
    print("Impossibile aprire la telecamera.\n")
    exit()

if video.get(cv.CAP_PROP_FRAME_WIDTH) > 1280:
    video.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
if video.get(cv.CAP_PROP_FRAME_HEIGHT) > 720:
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

video.set(cv.CAP_PROP_FPS, 30)

# Istanziamento di BlinkDetector
blink_detector = BlinkDetector()

tastiera = Controller()


def on_left_blink():
    print("Sbattuto Occhio SINISTRO\n")
    tastiera.press(Key.cmd_l)
    tastiera.release(Key.cmd_l)


def on_right_blink():
    print("Sbattuto Occhio DESTRO\n")


blink_detector.on_left_blink_callback = on_left_blink
blink_detector.on_right_blink_callback = on_right_blink


while True:
    status, frame = video.read()

    if not status:
        print("Errore, impossibile trovare un fotogramma.")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    blink_detector.frame_preparation(frame=frame, rgb=rgb_frame)

    # Display the resulting frame
    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
blink_detector.close()
video.release()
cv.destroyAllWindows()
