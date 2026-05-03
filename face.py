import face_recognition
import os
import numpy as np

KNOWN_FACES_DIR = "Known_faces"
FACE_TOLERANCE = 0.5

known_face_encodings = []
known_face_names = []

def load_faces():
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for file in os.listdir(KNOWN_FACES_DIR):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(KNOWN_FACES_DIR, file)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0])

load_faces()

def recognize_face(frame):
    if not known_face_encodings:
        return "Unknown"

    rgb = frame[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb)

    for face in encodings:
        distances = face_recognition.face_distance(known_face_encodings, face)
        best_match = np.argmin(distances)

        if distances[best_match] < FACE_TOLERANCE:
            return known_face_names[best_match]

    return "Unknown"
