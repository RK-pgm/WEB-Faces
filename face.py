import face_recognition
import os
import numpy as np

Known_face_Encoding = []
Known_face_names = []

def load_faces() :
    for file in os.listdir("Known_faces") :
        if file.endswith((".jpg", ".png")):
            path = f"Known_faces/{file}"
            img = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(img)
            if encoding :
                Known_face_Encoding.append(encoding[0])
                Known_face_names.append(file.split(".")[0])
load_faces()
def recognize_faces(frame) :
    rgb = frame [:,:,::-1]
    encodings = face_recognition.face_encodings(rgb)
    for face in encodings :
        distances = face_recognition.face_distance(Known_face_Encoding, face)
        match = np.argmin(distances)
        if distances[match] < 0.5 :
            return Known_face_names[match]
    return "Unknown"