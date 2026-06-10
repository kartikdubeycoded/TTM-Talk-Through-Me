import os
import shutil
import tensorflow as tf
import tensorflowjs as tfjs

MODEL_PATH = os.path.join("models", "alphabet.h5")
LABELS_PATH = os.path.join("models", "labels.json")
EXPORT_DIR = os.path.join("extension", "model")

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model file {MODEL_PATH} not found. Please run training/train_alphabet.py first.")
        return
        
    print(f"Loading Keras model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print(f"Exporting model to TensorFlow.js format in {EXPORT_DIR}...")
    if os.path.exists(EXPORT_DIR):
        # Clear existing to prevent duplicate weight shards
        shutil.rmtree(EXPORT_DIR)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Save the model
    tfjs.converters.save_keras_model(model, EXPORT_DIR)
    print("Model converted and saved successfully.")
    
    # Copy labels.json to extension/model directory for browser reference
    if os.path.exists(LABELS_PATH):
        shutil.copy(LABELS_PATH, os.path.join(EXPORT_DIR, "labels.json"))
        print(f"Copied labels mapping to {EXPORT_DIR}/labels.json")

if __name__ == "__main__":
    main()
