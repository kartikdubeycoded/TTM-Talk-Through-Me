import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from augment import augment_training_set

DATA_DIR = os.path.join("data", "alphabet")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "alphabet.h5")

def main():
    # Define classes: A-Z, SPACE, DELETE
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['SPACE', 'DELETE']
    
    X = []
    y = []
    
    print("Loading datasets...")
    for idx, cls_name in enumerate(classes):
        file_path = os.path.join(DATA_DIR, f"{cls_name}.npy")
        if not os.path.exists(file_path):
            print(f"Error: dataset file not found for '{cls_name}' at {file_path}")
            print("Please run pipeline/collect_data.py first.")
            return
            
        data = np.load(file_path)
        X.append(data)
        y.append(np.full(data.shape[0], idx))
        
    X = np.vstack(X).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    
    print(f"Dataset loaded. Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Augment AFTER the split, training half only — augmenting before the
    # split would leak near-duplicates into validation and inflate accuracy.
    X_train, y_train = augment_training_set(X_train, y_train, seed=42)
    print(f"Training set (augmented): {X_train.shape}, Validation set (raw): {X_val.shape}")
    
    # Build MLP Model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Train
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            )
        ]
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation complete. Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

    # Confusion matrix on the raw validation set: which letters get mistaken
    # for which. Saved as text so weak letters are visible at a glance.
    os.makedirs(MODEL_DIR, exist_ok=True)
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    cm = confusion_matrix(y_val, y_pred)
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.txt")
    with open(cm_path, "w") as f:
        f.write("rows = true class, cols = predicted class\n")
        f.write("     " + " ".join(f"{c:>4}" for c in classes) + "\n")
        for cls_name, row in zip(classes, cm):
            f.write(f"{cls_name:>4} " + " ".join(f"{v:>4}" for v in row) + "\n")
        f.write("\nTop confusions (true -> predicted: count):\n")
        off_diag = [(cm[i, j], classes[i], classes[j])
                    for i in range(len(classes)) for j in range(len(classes))
                    if i != j and cm[i, j] > 0]
        for count, true_c, pred_c in sorted(off_diag, reverse=True)[:10]:
            f.write(f"  {true_c} -> {pred_c}: {count}\n")
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model successfully saved to {MODEL_PATH}")

    # Write class labels to a text/json file for browser reference
    import json
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(classes, f)
    print(f"Labels mapping saved to {labels_path}")

if __name__ == "__main__":
    main()
