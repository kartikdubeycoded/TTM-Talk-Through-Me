import os
import sys
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
DATA_DIR = os.path.join("data", "alphabet")

def get_normalized_landmarks(hand_landmarks):
    wrist = hand_landmarks[0]
    mcp_9 = hand_landmarks[9]
    scale = math.sqrt(
        (mcp_9.x - wrist.x)**2 + 
        (mcp_9.y - wrist.y)**2 + 
        (mcp_9.z - wrist.z)**2
    )
    if scale == 0:
        scale = 1e-6

    normalized = []
    for lm in hand_landmarks:
        normalized.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale
        ])
    return normalized

def generate_synthetic_dataset():
    """
    Generates a synthetic landmark dataset for testing the pipeline
    in headless/automated environments.
    Creates 100 samples for each class A-Z, plus SPACE and DELETE (28 classes).
    """
    print("No webcam detected or headless mode requested.")
    print("Generating synthetic ASL landmark dataset for all 28 classes...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['SPACE', 'DELETE']
    np.random.seed(42)
    
    # Define distinct base shapes for each class
    for idx, cls_name in enumerate(classes):
        # Base shape: random hand-like coordinates
        base_shape = np.random.uniform(-0.5, 0.5, 63)
        # Anchor wrist at (0, 0, 0)
        base_shape[0:3] = 0.0
        # Knuckle 9 scale anchor
        base_shape[27:30] = [0.0, 0.5, 0.0]
        
        # Collect 150 samples with random spatial jittering (augmentation)
        samples = []
        for _ in range(150):
            # Jitter coordinates with small noise
            noise = np.random.normal(0, 0.03, 63)
            # Ensure wrist is always locked to 0
            noise[0:3] = 0.0
            sample = base_shape + noise
            samples.append(sample)
            
        samples = np.array(samples, dtype=np.float32)
        np.save(os.path.join(DATA_DIR, f"{cls_name}.npy"), samples)
        print(f"Saved synthetic dataset for '{cls_name}' with shape {samples.shape}")

    print("Synthetic dataset generation complete.")

def record_from_webcam(detector, cam_index=0):
    os.makedirs(DATA_DIR, exist_ok=True)
    cap = cv2.VideoCapture(cam_index)
    
    if not cap.isOpened():
        print(f"Failed to open webcam index {cam_index}.")
        generate_synthetic_dataset()
        return

    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['SPACE', 'DELETE']
    current_class_idx = 0

    print("\n--- Webcam Data Collection Active ---")
    print("Press 's' to start capturing 150 samples for the current letter.")
    print("Press 'n' to skip to the next letter.")
    print("Press 'q' to quit.")

    while current_class_idx < len(classes):
        cls_name = classes[current_class_idx]
        samples = []
        
        print(f"\nReady to collect landmarks for: '{cls_name}'")
        
        while len(samples) < 150:
            success, frame = cap.read()
            if not success:
                break
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Draw UI overlay
            cv2.putText(frame, f"Collect: {cls_name}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Progress: {len(samples)}/150", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press 's' to capture, 'n' next, 'q' quit", (30, h - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Detect hand
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            
            # Draw skeletal dots
            if detection_result.hand_landmarks:
                for hand_lms in detection_result.hand_landmarks:
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            cv2.imshow("SignToText - Collector", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Exiting data collection.")
                return
            elif key == ord('n'):
                print(f"Skipping letter '{cls_name}'")
                samples = []
                break
            elif key == ord('s') or len(samples) > 0:
                # Capture mode activated or in-progress
                if detection_result.hand_landmarks:
                    # Save normalized landmarks of the first hand
                    norm_lms = get_normalized_landmarks(detection_result.hand_landmarks[0])
                    samples.append(norm_lms)

        if len(samples) == 150:
            np.save(os.path.join(DATA_DIR, f"{cls_name}.npy"), np.array(samples))
            print(f"Saved {len(samples)} samples for '{cls_name}' to {DATA_DIR}/{cls_name}.npy")
            current_class_idx += 1
        else:
            current_class_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam data collection finished.")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not found. Running synthetic generator.")
        generate_synthetic_dataset()
        return

    # Check for headless flag or force-synthetic argument
    if len(sys.argv) > 1 and sys.argv[1] == "--synthetic":
        generate_synthetic_dataset()
        return

    print("Initializing detector for webcam checking...")
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    
    try:
        detector = vision.HandLandmarker.create_from_options(options)
        # Check if webcam can be opened, otherwise fallback to synthetic
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            generate_synthetic_dataset()
        else:
            cap.release()
            record_from_webcam(detector)
    except Exception as e:
        print(f"Error initializing MediaPipe/OpenCV: {e}. Falling back to synthetic generator.")
        generate_synthetic_dataset()

if __name__ == "__main__":
    main()
