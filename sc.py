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

# Store multiple selected faces
selected_faces = []

# Folder to save images
save_folder = "saved_faces"
os.makedirs(save_folder, exist_ok=True)

print("🟢 Press 's' to save a face (each press saves a new face).")
print("🟢 Press 'c' to save full image (with blur).")
print("🟣 Press 'q' to quit.")

def get_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def is_same_face(f1, f2):
    """Check if two faces are close to each other (same person)."""
    c1, c2 = get_center(f1), get_center(f2)
    dist = np.linalg.norm(np.array(c1) - np.array(c2))
    return dist < 80  # tolerance for small movement

def save_image(frame, name_prefix="capture"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_folder, f"{name_prefix}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    print(f"📸 Saved image: {filename}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)
    h, w, _ = frame.shape
    output = frame.copy()  # for display

    current_faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bbox.xmin * w), int(bbox.ymin * h), \
                           int(bbox.width * w), int(bbox.height * h)
            current_faces.append((x, y, bw, bh))

    # Blur all faces except selected ones
    for box in current_faces:
        should_blur = True
        for saved_face in selected_faces:
            if is_same_face(box, saved_face):
                should_blur = False
                break

        if should_blur:
            x, y, bw, bh = box
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            roi = output[y1:y2, x1:x2]
            if roi.size != 0:
                roi = cv2.GaussianBlur(roi, (99, 99), 30)
                output[y1:y2, x1:x2] = roi

    display_frame = output.copy()

    # Show rectangles only for live preview
    for sf in selected_faces:
        x, y, bw, bh = sf
        cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.imshow("🔒 Multi-Face Privacy Capture", display_frame)
    key = cv2.waitKey(1) & 0xFF

    # Save new face (add to list)
    if key == ord('s') and results.detections:
        largest = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
        bbox = largest.location_data.relative_bounding_box
        new_face = (int(bbox.xmin * w), int(bbox.ymin * h),
                    int(bbox.width * w), int(bbox.height * h))

        # Check if it's a new face (not near existing ones)
        if not any(is_same_face(new_face, sf) for sf in selected_faces):
            selected_faces.append(new_face)
            x, y, bw, bh = new_face
            face_crop = frame[y:y + bh, x:x + bw]
            save_image(face_crop, name_prefix=f"face_{len(selected_faces)}")
            print(f"✅ Face {len(selected_faces)} saved and stays unblurred.")
        else:
            print("⚠️ That face is already saved.")

    # Save full blurred image
    if key == ord('c'):
        save_image(output, name_prefix="full_capture")
        print("✅ Full image saved with blur.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
