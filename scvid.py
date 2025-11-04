import cv2
import mediapipe as mp
import numpy as np
import os
import datetime

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Start camera
cap = cv2.VideoCapture(0)

# List to store multiple saved faces
saved_faces_boxes = []

# Folder to save images and video
save_folder = "saved_faces"
os.makedirs(save_folder, exist_ok=True)

print("🟢 Press 's' to save your face (face crop only).")
print("🟢 Press 'c' to save full frame (with blur).")
print("🟣 Press 'v' to start/stop video recording.")
print("🟣 Press 'q' to quit.")

def get_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def is_same_face(f1, f2):
    c1, c2 = get_center(f1), get_center(f2)
    dist = np.linalg.norm(np.array(c1) - np.array(c2))
    return dist < 80  # tolerance for movement

def save_image(frame, name_prefix="capture"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"{name_prefix}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Saved image: {filename}")

def init_video_writer(frame, prefix="video"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"{prefix}_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = frame.shape
    out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))
    print(f"🎬 Video recording started: {filename}")
    return out

video_writer = None
recording = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    h, w, _ = frame.shape
    output = frame.copy()  # for display

    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), \
                           int(bbox.width * w), int(bbox.height * h)
            faces.append((x, y, bw, bh))

        # Blur non-selected faces
        for box in faces:
            if any(is_same_face(box, saved) for saved in saved_faces_boxes):
                continue  # skip blur for all saved faces
            x, y, bw, bh = box
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            roi = output[y1:y2, x1:x2]
            if roi.size != 0:
                roi = cv2.GaussianBlur(roi, (99, 99), 30)
                output[y1:y2, x1:x2] = roi

    # Draw rectangles for live display
    display_frame = output.copy()
    for box in saved_faces_boxes:
        x, y, bw, bh = box
        cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.imshow("🔒 Face Privacy Video", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # Save face crop (add to saved faces)
    if key == ord('s') and faces:
        largest = max(faces, key=lambda b: b[2]*b[3])
        saved_faces_boxes.append(largest)
        x, y, bw, bh = largest
        face_crop = frame[y:y + bh, x:x + bw]
        save_image(face_crop, name_prefix=f"your_face_{len(saved_faces_boxes)}")
        print(f"✅ Face {len(saved_faces_boxes)} saved and will stay unblurred.")

    # Save full frame with blur
    if key == ord('c'):
        save_image(output, name_prefix="full_capture")
        print("✅ Full frame saved with blur.")

    # Start/stop video recording
    if key == ord('v'):
        if not recording:
            video_writer = init_video_writer(output)
            recording = True
        else:
            video_writer.release()
            recording = False
            print("🎬 Video recording stopped.")

    # Write current frame if recording
    if recording and video_writer is not None:
        video_writer.write(output)

    elif key == ord('q'):
        break

if video_writer is not None:
    video_writer.release()
cap.release()
cv2.destroyAllWindows()
