import cv2 as cv

from mio.blinkDetector import BlinkDetector

video = cv.VideoCapture(1)

if not video.isOpened():
    print("Impossibile aprire la telecamera.\n")
    exit()

blink_detector = BlinkDetector()  # Istanziamento di BlinkDetector

while True:
    status, frame = video.read()

    if not status:
        print("Errore, impossibile trovare un fotogramma.")
        break

    blink_detector.process_frame(frame=frame)

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow("frame", gray)
    if cv.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
video.release()
cv.destroyAllWindows()
