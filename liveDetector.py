# import essential modules

import numpy
import cv2


def convertToGRAY(image):
    """Converts image to grayscale."""
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def widthHeightDividedBy(image, value):
    """Divides image's width & height by a value (for further use)."""
    
    w, h = image.shape[:2]
    return int(w/value), int(h/value)


def detector(frame, win_name):
    """Detects faces."""

    gray = convertToGRAY(frame)
    minSize = widthHeightDividedBy(gray, 8)
    faces = face_classifier.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, minSize)
    print("Faces found: ", len(faces))

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow(win_name, frame)


camera = cv2.VideoCapture(0)
cv2.namedWindow("Detector", cv2.WINDOW_NORMAL)
face_classifier = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")  #  loads face cascades.

if camera.isOpened():  #  checks whether camera is opened or not.
    while True:
        ret, frame = camera.read()
        detector(frame, "Detector")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("Camera is not opening, try again.")

camera.release()  # don't forget to do this.
cv2.destroyAllWindows()
