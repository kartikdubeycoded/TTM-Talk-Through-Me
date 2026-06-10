import os
import sys
import urllib.request
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model URL and Path
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)

def get_normalized_landmarks(hand_landmarks):
    """
    Applies Coordinate Anchoring (wrist as 0,0,0) and Scaling
    (divided by distance between wrist and middle finger knuckle).
    Returns a flattened array of 63 values (21 points * 3 coordinates).
    """
    wrist = hand_landmarks[0]
    mcp_9 = hand_landmarks[9]  # Middle finger knuckle
    
    # Calculate scale factor (Euclidean distance between wrist and middle knuckle)
    scale = math.sqrt(
        (mcp_9.x - wrist.x)**2 + 
        (mcp_9.y - wrist.y)**2 + 
        (mcp_9.z - wrist.z)**2
    )
    if scale == 0:
        scale = 1e-6

    normalized = []
    for lm in hand_landmarks:
        # Subtract wrist coordinates and divide by scale factor
        normalized.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])
    return normalized

def draw_hand_skeleton(image, hand_landmarks):
    """
    Draw skeleton connections on the image using OpenCV.
    """
    h, w, _ = image.shape
    
    # Convert normalized landmarks to pixel coordinates
    coords = []
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        # Draw joint circle
        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

    # Connections definition
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17)
    ]

    for start, end in connections:
        if start < len(coords) and end < len(coords):
            cv2.line(image, coords[start], coords[end], (255, 0, 0), 2)

def main():
    download_model()

    print("Initializing MediaPipe Hand Landmarker...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Allow custom camera index from command line
    cam_index = 0
    if len(sys.argv) > 1:
        try:
            cam_index = int(sys.argv[1])
        except ValueError:
            pass

    print(f"Opening webcam (index {cam_index})...")
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        print(f"Could not open webcam index {cam_index}.")
        print("Starting in headless mode for model prediction testing.")
        
        # Test with a mock image loop
        print("Creating a mock black image to test pipeline integration...")
        mock_img = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mock_img)
        detection_result = detector.detect(mp_image)
        print("Headless mode test complete. Detector runs successfully.")
        return

    print("Tracking active. Press 'q' in the window to quit.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break

        # Flip horizontally for natural mirror view
        frame = cv2.flip(frame, 1)

        # Convert to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run hand landmark detection
        detection_result = detector.detect(mp_image)

        # Process results
        if detection_result.hand_landmarks:
            for i, hand_lms in enumerate(detection_result.hand_landmarks):
                # Draw skeleton
                draw_hand_skeleton(frame, hand_lms)
                
                # Get normalized coords
                norm_coords = get_normalized_landmarks(hand_lms)
                
                # Print a small subset of values to show it's working
                print(f"Hand {i} detected. Wrist scale={round(norm_coords[3], 3)}")

        cv2.imshow("SignToText - Landmark Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")

if __name__ == "__main__":
    main()
