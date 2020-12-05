# freeCodeCamp OpenCV Course - Full Tutorial with Python
# https://github.com/ageitgey/face_recognition

import cv2 as cv
import face_recognition

video_capture = cv.VideoCapture(0, cv.CAP_DSHOW)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_names = []


def haar_cascades(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=21)

    for (x, y, w, h) in faces_rect:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Haar Cascades', img)


def face_recognition_dlib(img):
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)

    for (top, right, bottom, left) in face_locations:
        cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    cv.imshow('Face Recognition dlib', img)


while True:
    ret, frame = video_capture.read()
    small_frame = cv.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    haar_results = small_frame.copy()
    face_recognition_results = small_frame.copy()

    haar_cascades(haar_results)
    face_recognition_dlib(face_recognition_results)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
