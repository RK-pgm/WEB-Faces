import face_recognition
import os
import cv2
import numpy as np

Known_face_Encoding = []
Known_face_names = []

def load_faces():
    Known_face_Encoding.clear()
    Known_face_names.clear()
    folder = "Known_faces"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)
            try:
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    Known_face_Encoding.append(encoding[0])
                    Known_face_names.append(os.path.splitext(file)[0])
                    print(f"[Load] โหลดใบหน้า: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")

load_faces()

def reload_faces():
    load_faces()

def recognize_face(frame):
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    rgb = frame[:, :, ::-1]
    encodings = face_recognition.face_encodings(rgb)

    if not encodings:
        return "Unknown"

    for face in encodings:
        if len(Known_face_Encoding) == 0:
            return "Unknown"
        distances = face_recognition.face_distance(Known_face_Encoding, face)
        best_match = np.argmin(distances)
        print(f"[Scan] distance={distances[best_match]:.3f}")
        if distances[best_match] < 0.6:
            return Known_face_names[best_match]

    return "Unknown"