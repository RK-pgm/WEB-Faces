from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
from face import recognize_face, reload_faces

app = Flask(__name__, template_folder=".")
DB_PATH = "database.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, student_id TEXT, timestamp TEXT)")
        conn.commit()

def log_attendance(name, student_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO attendance (name, student_id, timestamp) VALUES (?, ?, ?)", (name, student_id, now))
            conn.commit()
    except Exception as e:
        print(f"Database Error: {e}")

def decode_image(image_data):
    try:
        header, encoded = image_data.split(",", 1)
        img_data = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scan_page")
def scan_page():
    return render_template("scan.html")

@app.route("/register_page")
def register_page():
    return render_template("register.html")

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    student_id = data.get("id")
    image_b64 = data.get("image")

    if not name or not student_id or not image_b64:
        return jsonify({"error": "กรุณากรอกข้อมูลให้ครบถ้วน"}), 400

    safe_name = "".join(ch for ch in name if ch.isalnum() or ch in (" ", "-", "_")).strip()
    if not safe_name:
        return jsonify({"error": "Invalid name"}), 400

    frame = decode_image(image_b64)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    os.makedirs("Known_faces", exist_ok=True)
    # Windows ไม่รองรับเครื่องหมาย | ในชื่อไฟล์ จึงเปลี่ยนเป็น _ID_
    path = os.path.join("Known_faces", f"{safe_name}_ID_{student_id}.jpg")
    cv2.imwrite(path, frame)
    reload_faces()
    return jsonify({"message": f"ลงทะเบียนคุณ {safe_name} เรียบร้อยแล้ว", "name": safe_name})

@app.route("/scan", methods=["POST"])
def scan():
    data = request.get_json()
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Missing image"}), 400
    frame = decode_image(image_b64)
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400
    result = recognize_face(frame)
    if result != "Unknown":
        if "_ID_" in result:
            name, student_id = result.split("_ID_", 1)
        else:
            parts = result.rsplit("_", 1)
            name = parts[0] if len(parts) > 1 else result
            student_id = parts[1] if len(parts) > 1 else "Unknown"
        log_attendance(name, student_id)
        return jsonify({"name": name, "id": student_id})
    return jsonify({"name": "Unknown"})

@app.route("/dashboard")
def dashboard():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT name, student_id, timestamp FROM attendance ORDER BY timestamp DESC")
        data = c.fetchall()
    return render_template("dashboard.html", data=data)

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)