# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load YOLO model
# model = YOLO("Model/ppe.pt")  # Update with your model path

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     logs = []  # Store logs to send to frontend

#     if 'file' not in request.files:
#         logs.append("❌ No file received!")
#         return jsonify({"logs": logs, "error": "No file part"}), 400

#     file = request.files['file']
    
#     if file.filename == '':
#         logs.append("❌ Empty filename!")
#         return jsonify({"logs": logs, "error": "No selected file"}), 400

#     logs.append(f"✅ Received file: {file.filename}")

#     try:
#         img = Image.open(file.stream)
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         logs.append("✅ Image successfully loaded and converted!")
#     except Exception as e:
#         logs.append(f"❌ Error loading image: {str(e)}")
#         return jsonify({"logs": logs, "error": "Failed to load image"}), 400

#     # Perform YOLO inference
#     results = model(img)

#     if len(results) == 0 or len(results[0].boxes) == 0:
#         logs.append("❌ No detections found!")
#         return jsonify({"logs": logs, "message": "No objects detected"}), 200

#     logs.append(f"✅ Raw detections: {len(results[0].boxes)} object(s) detected!")

#     boxes = results[0].boxes
#     classes = results[0].names if hasattr(results[0], 'names') else model.names
#     detected_info = []
#     class_counts = {}

#     for box in boxes:
#         class_id = int(box.cls[0].item())
#         label = classes[class_id]
#         confidence = float(box.conf[0].item())
        
#         detected_info.append(f"{label} ({confidence:.2f})")
        
#         if label in class_counts:
#             class_counts[label] += 1
#         else:
#             class_counts[label] = 1

#     logs.append("🎯 Detected objects with confidence:")
#     logs.extend([f"• {info}" for info in detected_info])

#     logs.append("🔢 Count of each class:")
#     logs.extend([f"• {label}: {count}" for label, count in class_counts.items()])

#     return jsonify({"logs": logs, "message": "Detection completed"}), 200


# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load YOLO model
# model = YOLO("Model/ppe.pt")  # Update with your model path

# # New: Global variable for receiver email
# receiver_email = None

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     logs = []  # Store logs to send to frontend

#     if 'file' not in request.files:
#         logs.append("❌ No file received!")
#         return jsonify({"logs": logs, "error": "No file part"}), 400

#     file = request.files['file']
    
#     if file.filename == '':
#         logs.append("❌ Empty filename!")
#         return jsonify({"logs": logs, "error": "No selected file"}), 400

#     logs.append(f"✅ Received file: {file.filename}")

#     try:
#         img = Image.open(file.stream)
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         logs.append("✅ Image successfully loaded and converted!")
#     except Exception as e:
#         logs.append(f"❌ Error loading image: {str(e)}")
#         return jsonify({"logs": logs, "error": "Failed to load image"}), 400

#     # Perform YOLO inference
#     results = model(img)

#     if len(results) == 0 or len(results[0].boxes) == 0:
#         logs.append("❌ No detections found!")
#         return jsonify({"logs": logs, "message": "No objects detected"}), 200

#     logs.append(f"✅ Raw detections: {len(results[0].boxes)} object(s) detected!")

#     boxes = results[0].boxes
#     classes = results[0].names if hasattr(results[0], 'names') else model.names
#     detected_info = []
#     class_counts = {}

#     for box in boxes:
#         class_id = int(box.cls[0].item())
#         label = classes[class_id]
#         confidence = float(box.conf[0].item())
        
#         detected_info.append(f"{label} ({confidence:.2f})")
        
#         if label in class_counts:
#             class_counts[label] += 1
#         else:
#             class_counts[label] = 1

#     logs.append("🎯 Detected objects with confidence:")
#     logs.extend([f"• {info}" for info in detected_info])

#     logs.append("🔢 Count of each class:")
#     logs.extend([f"• {label}: {count}" for label, count in class_counts.items()])

#     return jsonify({"logs": logs, "message": "Detection completed"}), 200

# # New: Route to receive receiver email after Google Sign In
# @app.route('/set-receiver-email', methods=['POST'])
# def set_receiver_email():
#     global receiver_email
#     data = request.json
#     receiver_email = data.get('email')
#     print(f"✅ Receiver email updated to: {receiver_email}")
#     return {"message": "Receiver email updated", "receiver_email": receiver_email}

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify, Response
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import Image
# from flask_cors import CORS
# import os
# import subprocess

# app = Flask(__name__)
# CORS(app, origins="http://localhost:3000", supports_credentials=True)  # Enhanced CORS settings

# webcam_process = None
# camera = None

# # Video camera class to handle frame processing
# class VideoCamera:
#     def __init__(self):
#         self.video = cv2.VideoCapture(0)
#         self.model = YOLO("Model/ppe.pt")
        
#     def __del__(self):
#         self.video.release()
        
#     def get_frame(self):
#         success, frame = self.video.read()
#         if not success:
#             return None
            
#         # Process frame with YOLO
#         results = self.model(frame)
#         annotated_frame = results[0].plot()
        
#         # Encode the processed frame
#         ret, jpeg = cv2.imencode('.jpg', annotated_frame)
#         return jpeg.tobytes()

# # Load model
# try:
#     model = YOLO("Model/ppe.pt")
#     print("✅ YOLO model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {str(e)}")

# @app.route('/set-receiver-email', methods=['POST'])
# def set_receiver_email():
#     data = request.get_json()
#     email = data.get("email")

#     if not email:
#         return jsonify({"message": "Email is required"}), 400

#     # Save email to file
#     with open(".receiver_email.txt", "w") as f:
#         f.write(email)

#     print(f"📩 Receiver email set to: {email}")
#     return jsonify({"message": f"Receiver email set to {email}"}), 200

# def get_receiver_email():
#     """Utility function to read the latest receiver email"""
#     if os.path.exists(".receiver_email.txt"):
#         with open(".receiver_email.txt", "r") as f:
#             return f.read().strip()
#     return None

# # Generate frames for video streaming
# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         if frame is None:
#             break
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#     """Route to stream the webcam feed with YOLO detections"""
#     global camera
#     # Initialize the camera if not already done
#     if camera is None:
#         camera = VideoCamera()
#     # Return a streaming response
#     return Response(gen(camera),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/start-webcam', methods=['POST'])
# def start_webcam():
#     global webcam_process, camera
#     logs = []

#     # Option 1: Use the integrated approach
#     try:
#         if camera is None:
#             camera = VideoCamera()
#         logs.append("📹 Starting webcam...")
#         logs.append("✅ webcam.py started.")
#     except Exception as e:
#         logs.append(f"❌ Error starting camera: {str(e)}")
#         return jsonify({"logs": logs, "error": "Failed to start camera"}), 500

#     # Option 2: Keep the subprocess approach if needed
#     # if webcam_process is None or webcam_process.poll() is not None:
#     #     webcam_process = subprocess.Popen(['python', 'webcam.py'])
#     #     logs.append("✅ webcam.py started.")
#     # else:
#     #     logs.append("⚠ webcam.py is already running.")

#     return jsonify({"logs": logs}), 200

# @app.route('/stop-webcam', methods=['POST'])
# def stop_webcam():
#     global webcam_process, camera
#     logs = []

#     # Option 1: Release the camera if using integrated approach
#     if camera is not None:
#         try:
#             camera.__del__()
#             camera = None
#             logs.append("🛑 Camera released.")
#         except Exception as e:
#             logs.append(f"❌ Error stopping camera: {str(e)}")
    
#     # Option 2: Terminate the subprocess if using that approach
#     if webcam_process and webcam_process.poll() is None:
#         webcam_process.terminate()
#         logs.append("🛑 webcam.py process terminated.")
#         webcam_process = None
#     else:
#         logs.append("⚠ No active webcam process to stop.")

#     return jsonify({"logs": logs}), 200

# @app.route('/upload', methods=['POST'])
# def upload_image():
#     logs = []

#     if 'file' not in request.files:
#         logs.append("❌ No file received!")
#         return jsonify({"logs": logs, "error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         logs.append("❌ Empty filename!")
#         return jsonify({"logs": logs, "error": "No selected file"}), 400

#     logs.append(f"✅ Received file: {file.filename}")

#     try:
#         img = Image.open(file.stream).convert('RGB')
#         img = np.array(img)
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         logs.append("✅ Image loaded and converted.")
#     except Exception as e:
#         logs.append(f"❌ Image load error: {str(e)}")
#         return jsonify({"logs": logs, "error": "Failed to load image"}), 400

#     try:
#         results = model(img)[0]
#         boxes = results.boxes
#         class_names = model.names
#     except Exception as e:
#         logs.append(f"❌ Model inference failed: {str(e)}")
#         return jsonify({"logs": logs, "error": "Inference failed"}), 500

#     if boxes is None or len(boxes) == 0:
#         logs.append("❌ No detections.")
#         return jsonify({"logs": logs, "message": "No objects detected"}), 200

#     logs.append(f"✅ Detected {len(boxes)} object(s)")

#     detected_info = []
#     class_counts = {}

#     for box in boxes:
#         class_id = int(box.cls.item())
#         label = class_names[class_id]
#         confidence = float(box.conf.item())
#         detected_info.append(f"{label} ({confidence:.2f})")
#         class_counts[label] = class_counts.get(label, 0) + 1

#     logs.extend(["🎯 Detected objects:"] + [f"• {info}" for info in detected_info])
#     logs.extend(["🔢 Count of each class:"] + [f"• {label}: {count}" for label, count in class_counts.items()])

#     # Example usage of dynamic email
#     receiver_email = get_receiver_email()
#     logs.append(f"📩 Alert would be sent to: {receiver_email}")

#     return jsonify({"logs": logs, "message": "Detection completed"}), 200

# if __name__ == '__main__':
#     app.run(debug=True, host="0.0.0.0", port=5000)


from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from flask_cors import CORS
import os
import subprocess

app = Flask(__name__)
CORS(app, origins="http://localhost:3000", supports_credentials=True)

webcam_process = None
camera = None

# Video camera class to handle frame processing
class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.model = YOLO("Model/ppe.pt")
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        results = self.model(frame)
        annotated_frame = results[0].plot()
        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        return jpeg.tobytes()

# Load model
try:
    model = YOLO("Model/ppe.pt")
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")

@app.route('/set-receiver-email', methods=['POST'])
def set_receiver_email():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"message": "Email is required"}), 400

    with open(".receiver_email.txt", "w") as f:
        f.write(email)

    print(f"📩 Receiver email set to: {email}")
    return jsonify({"message": f"Receiver email set to {email}"}), 200

def get_receiver_email():
    if os.path.exists(".receiver_email.txt"):
        with open(".receiver_email.txt", "r") as f:
            return f.read().strip()
    return None

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global camera
    if camera is None:
        camera = VideoCamera()
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start-webcam', methods=['POST'])
def start_webcam():
    global webcam_process, camera
    logs = []

    try:
        if camera is None:
            camera = VideoCamera()
        logs.append("📹 Starting webcam...")
        logs.append("✅ Webcam initialized.")
    except Exception as e:
        logs.append(f"❌ Error starting camera: {str(e)}")
        return jsonify({"logs": logs, "error": "Failed to start camera"}), 500

    return jsonify({"logs": logs}), 200

@app.route('/stop-webcam', methods=['POST'])
def stop_webcam():
    global webcam_process, camera
    logs = []

    if camera is not None:
        try:
            camera.__del__()
            camera = None
            logs.append("🛑 Camera released.")
        except Exception as e:
            logs.append(f"❌ Error stopping camera: {str(e)}")
    
    if webcam_process and webcam_process.poll() is None:
        webcam_process.terminate()
        logs.append("🛑 webcam.py process terminated.")
        webcam_process = None
    else:
        logs.append("⚠ No active webcam process to stop.")

    return jsonify({"logs": logs}), 200

@app.route('/upload', methods=['POST'])
def upload_image():
    logs = []

    if 'file' not in request.files:
        logs.append("❌ No file received!")
        return jsonify({"logs": logs, "error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        logs.append("❌ Empty filename!")
        return jsonify({"logs": logs, "error": "No selected file"}), 400

    logs.append(f"✅ Received file: {file.filename}")

    try:
        img = Image.open(file.stream).convert('RGB')
        img = np.array(img)
        logs.append(f"ℹ️ Image shape: {img.shape}, dtype: {img.dtype}")
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Invalid image shape")
    except Exception as e:
        logs.append(f"❌ Image load error: {str(e)}")
        return jsonify({"logs": logs, "error": "Failed to load image"}), 400

    try:
        results = model(img)[0]
        boxes = results.boxes
        class_names = model.names
    except Exception as e:
        logs.append(f"❌ Model inference failed: {str(e)}")
        return jsonify({"logs": logs, "error": "Inference failed"}), 500

    if boxes is None or len(boxes) == 0:
        logs.append("⚠ No detections.")
        return jsonify({"logs": logs, "message": "No objects detected"}), 200

    logs.append(f"✅ Detected {len(boxes)} object(s)")

    detected_info = []
    class_counts = {}

    for box in boxes:
        try:
            class_id = int(box.cls.item())
            label = class_names[class_id]
            confidence = float(box.conf.item())
            detected_info.append(f"{label} ({confidence:.2f})")
            class_counts[label] = class_counts.get(label, 0) + 1
        except Exception as e:
            logs.append(f"⚠ Error processing box: {e}")
            continue

    logs.extend(["🎯 Detected objects:"] + [f"• {info}" for info in detected_info])
    logs.extend(["🔢 Count of each class:"] + [f"• {label}: {count}" for label, count in class_counts.items()])

    receiver_email = get_receiver_email()
    logs.append(f"📩 Alert would be sent to: {receiver_email}")

    return jsonify({"logs": logs, "message": "Detection completed"}), 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
