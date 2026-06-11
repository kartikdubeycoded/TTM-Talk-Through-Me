import os
import shutil
import urllib.request

ASSETS = {
    "extension/lib/tf.min.js": "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js",
    "extension/lib/vision_bundle.js": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/vision_bundle.mjs",
    "extension/wasm/vision_wasm_internal.js": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm/vision_wasm_internal.js",
    "extension/wasm/vision_wasm_internal.wasm": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm/vision_wasm_internal.wasm",
    # no-SIMD fallback: Brave hides CPU features for fingerprint protection,
    # so MediaPipe requests this variant there. Without it: instant Init Failed.
    "extension/wasm/vision_wasm_nosimd_internal.js": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm/vision_wasm_nosimd_internal.js",
    "extension/wasm/vision_wasm_nosimd_internal.wasm": "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm/vision_wasm_nosimd_internal.wasm"
}

def download_assets():
    for local_path, url in ASSETS.items():
        dir_name = os.path.dirname(local_path)
        os.makedirs(dir_name, exist_ok=True)
        
        print(f"Downloading {url} -> {local_path}...")
        try:
            urllib.request.urlretrieve(url, local_path)
            print("Download successful.")
        except Exception as e:
            print(f"Error downloading {url}: {e}")

    # Copy hand_landmarker.task to extension/model
    os.makedirs("extension/model", exist_ok=True)
    if os.path.exists("hand_landmarker.task"):
        print("Copying hand_landmarker.task to extension/model/hand_landmarker.task...")
        shutil.copy("hand_landmarker.task", "extension/model/hand_landmarker.task")
        print("Copy complete.")
    else:
        print("Warning: hand_landmarker.task not found in root workspace. Please run pipeline/track_hands.py first.")

if __name__ == "__main__":
    download_assets()
