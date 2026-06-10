// content.js - SignToText Captions Overlay Engine

let extensionEnabled = false;
let chatInjectionEnabled = false;
let confidenceThreshold = 0.90;
let spacingDelay = 750;

let visionModule = null;
let landmarker = null;
let tfModel = null;
let labelMap = [];

let webcamStream = null;
let videoElement = null;
let canvasElement = null;
let canvasCtx = null;
let trackingLoopActive = false;

// UI Elements
let overlayElement = null;
let statusDot = null;
let bufferText = null;
let sentenceContainer = null;
let sentencePlaceholder = null;

// NLP / Sentence States
let currentSentence = "";
let currentWordBuffer = "";
let lastPredictedIndex = -1;
let sameFrameCount = 0;
const LOCK_IN_FRAMES = 5; // Lock-in a letter after 5 consecutive matching frames

// Temporal Spacing States
let lastPredictionTime = 0;
let lastWristPosition = null;
let wristVelocityRolling = 0;
const VELOCITY_SPACE_THRESHOLD = 0.005; // Wrist movement near zero

// Listen for messages from background/popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "settingsChanged") {
    const changes = request.changes;
    if (changes.extensionEnabled) {
      handleStateToggle(changes.extensionEnabled.newValue);
    }
    if (changes.chatInjectionEnabled) {
      chatInjectionEnabled = changes.chatInjectionEnabled.newValue;
    }
    if (changes.detectionThreshold) {
      confidenceThreshold = changes.detectionThreshold.newValue;
    }
    if (changes.spacingDelay) {
      spacingDelay = changes.spacingDelay.newValue;
    }
  }
});

// Load initial settings
chrome.storage.local.get([
  "extensionEnabled",
  "chatInjectionEnabled",
  "detectionThreshold",
  "spacingDelay"
], (data) => {
  extensionEnabled = data.extensionEnabled || false;
  chatInjectionEnabled = data.chatInjectionEnabled || false;
  confidenceThreshold = data.detectionThreshold || 0.90;
  spacingDelay = data.spacingDelay || 750;

  if (extensionEnabled) {
    initSignToText();
  }
});

function handleStateToggle(enabled) {
  extensionEnabled = enabled;
  if (enabled) {
    initSignToText();
  } else {
    stopSignToText();
  }
}

async function initSignToText() {
  if (overlayElement) return; // Already running

  createOverlayDOM();
  updateStatusUI("Initializing...", "lost");

  try {
    // 1. Dynamically import MediaPipe Vision module
    if (!visionModule) {
      const visionPath = chrome.runtime.getURL("lib/vision_bundle.js");
      visionModule = await import(visionPath);
    }

    // 2. Initialize MediaPipe Hand Landmarker
    if (!landmarker) {
      const wasmPath = chrome.runtime.getURL("wasm");
      const vision = await visionModule.FilesetResolver.forVisionTasks(wasmPath);
      
      const modelPath = chrome.runtime.getURL("model/hand_landmarker.task");
      landmarker = await visionModule.HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
      });
    }

    // 3. Build TFJS model from exported weights
    // (plain-JSON export — see training/export_weights.py for why we
    // don't use the tensorflowjs converter format)
    if (!tfModel) {
      const weightsPath = chrome.runtime.getURL("model/weights.json");
      const weightsResponse = await fetch(weightsPath);
      const spec = await weightsResponse.json();

      tfModel = tf.sequential();
      spec.layers.forEach((layer, i) => {
        tfModel.add(tf.layers.dense({
          units: layer.units,
          activation: layer.activation,
          inputShape: i === 0 ? [spec.inputSize] : undefined
        }));
      });
      tfModel.setWeights(
        spec.layers.flatMap(l => [tf.tensor2d(l.kernel), tf.tensor1d(l.bias)])
      );

      labelMap = spec.labels;
      console.log("Loaded alphabet vocabulary:", labelMap);
    }

    // 4. Initialize Webcam
    videoElement = document.createElement("video");
    videoElement.style.display = "none";
    document.body.appendChild(videoElement);

    canvasElement = document.createElement("canvas");
    canvasCtx = canvasElement.getContext("2d");

    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: { ideal: 30 } }
    });
    
    videoElement.srcObject = webcamStream;
    videoElement.play();

    videoElement.onloadedmetadata = () => {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      startTrackingLoop();
    };

  } catch (err) {
    console.error("SignToText Initialization Error:", err);
    updateStatusUI("Init Failed", "lost");
    showErrorMessage(err.message || "Failed to start camera or models.");
  }
}

function stopSignToText() {
  trackingLoopActive = false;
  
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  
  if (videoElement) {
    videoElement.remove();
    videoElement = null;
  }

  if (overlayElement) {
    overlayElement.remove();
    overlayElement = null;
  }

  // Clear states
  currentSentence = "";
  currentWordBuffer = "";
  lastPredictedIndex = -1;
  sameFrameCount = 0;
  lastWristPosition = null;
}

function startTrackingLoop() {
  trackingLoopActive = true;
  updateStatusUI("Ready", "lost");
  requestAnimationFrame(processFrame);
}

async function processFrame() {
  if (!trackingLoopActive) return;

  if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
    // Draw current webcam frame onto hidden canvas
    canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
    const imageData = canvasCtx.getImageData(0, 0, canvasElement.width, canvasElement.height);
    
    // Pass image data to MediaPipe HandLandmarker
    const startTimeMs = performance.now();
    const result = landmarker.detectForVideo(imageData, startTimeMs);

    // NOTE: the JS tasks-vision API names this `landmarks`
    // (the Python API calls it `hand_landmarks`)
    if (result.landmarks && result.landmarks.length > 0) {
      updateStatusUI("Tracking", "tracking");

      const handLms = result.landmarks[0];
      const normLms = getNormalizedLandmarks(handLms);
      
      // Track wrist velocity (displacement)
      trackWristVelocity(handLms[0]);

      // Run inference
      runInference(normLms);
    } else {
      updateStatusUI("No Hands", "lost");
      // Handle hand-loss spacing boundary
      handleHandLossSpacing();
    }
  }

  requestAnimationFrame(processFrame);
}

function trackWristVelocity(wrist) {
  if (lastWristPosition) {
    const dist = Math.sqrt(
      Math.pow(wrist.x - lastWristPosition.x, 2) +
      Math.pow(wrist.y - lastWristPosition.y, 2) +
      Math.pow(wrist.z - lastWristPosition.z, 2)
    );
    // Smooth velocity with rolling average
    wristVelocityRolling = (wristVelocityRolling * 0.7) + (dist * 0.3);
  }
  lastWristPosition = { x: wrist.x, y: wrist.y, z: wrist.z };
}

function getNormalizedLandmarks(landmarks) {
  const wrist = landmarks[0];
  const mcp_9 = landmarks[9];
  const scale = Math.sqrt(
    Math.pow(mcp_9.x - wrist.x, 2) +
    Math.pow(mcp_9.y - wrist.y, 2) +
    Math.pow(mcp_9.z - wrist.z, 2)
  );
  
  const factor = scale === 0 ? 1e-6 : scale;
  const normalized = [];
  
  for (const lm of landmarks) {
    normalized.push(
      (lm.x - wrist.x) / factor,
      (lm.y - wrist.y) / factor,
      (lm.z - wrist.z) / factor
    );
  }
  return normalized;
}

function runInference(normLms) {
  tf.tidy(() => {
    const tensor = tf.tensor2d([normLms]);
    const predictions = tfModel.predict(tensor);
    const scores = predictions.dataSync();
    
    const maxIdx = predictions.argMax(-1).dataSync()[0];
    const confidence = scores[maxIdx];

    if (confidence >= confidenceThreshold) {
      if (maxIdx === lastPredictedIndex) {
        sameFrameCount++;
        if (sameFrameCount === LOCK_IN_FRAMES) {
          // Lock-in predicted class
          const character = labelMap[maxIdx];
          handleLockedPrediction(character);
        }
      } else {
        lastPredictedIndex = maxIdx;
        sameFrameCount = 0;
      }
    }
  });
}

function handleLockedPrediction(character) {
  const now = Date.now();
  
  // Debounce to prevent multiple character logs from a single hold
  if (now - lastPredictionTime < 1200) {
    return;
  }
  
  lastPredictionTime = now;

  if (character === "SPACE") {
    appendSpace();
  } else if (character === "DELETE") {
    handleBackspace();
  } else {
    // Append to current word buffer
    currentWordBuffer += character;
    updateUIBuffer(currentWordBuffer);
    
    // Auto-spacing check: if hand is stationary after signing a letter
    if (wristVelocityRolling < VELOCITY_SPACE_THRESHOLD) {
      setTimeout(() => {
        if (Date.now() - lastPredictionTime >= spacingDelay && currentWordBuffer.length > 0) {
          commitWord();
        }
      }, spacingDelay);
    }
  }
}

function handleHandLossSpacing() {
  if (currentWordBuffer.length > 0 && Date.now() - lastPredictionTime >= spacingDelay) {
    commitWord();
  }
}

function commitWord() {
  if (currentWordBuffer.length === 0) return;
  
  currentSentence += (currentSentence.length > 0 ? " " : "") + currentWordBuffer;
  currentWordBuffer = "";
  
  updateUISentence(currentSentence);
  updateUIBuffer("");
  
  if (chatInjectionEnabled) {
    injectIntoChat(currentSentence);
  }
}

function appendSpace() {
  commitWord();
  currentSentence += " ";
  updateUISentence(currentSentence);
}

function handleBackspace() {
  if (currentWordBuffer.length > 0) {
    currentWordBuffer = currentWordBuffer.slice(0, -1);
    updateUIBuffer(currentWordBuffer);
  } else if (currentSentence.length > 0) {
    // Backspace last word in sentence
    const words = currentSentence.trim().split(" ");
    words.pop();
    currentSentence = words.join(" ");
    updateUISentence(currentSentence);
  }
}

// Injects translated sentence into Google Meet / Teams chat inputs
function injectIntoChat(text) {
  const chatInput = document.querySelector('textarea[aria-label="Send a message to everyone"]') ||
                    document.querySelector('textarea[placeholder="Type message here..."]') ||
                    document.querySelector('textarea[aria-label="Type a new message"]');
  if (chatInput) {
    chatInput.value = text;
    chatInput.dispatchEvent(new Event('input', { bubbles: true }));
  }
}

// UI Rendering Functions
function createOverlayDOM() {
  overlayElement = document.createElement("div");
  overlayElement.id = "signtotext-overlay";
  overlayElement.innerHTML = `
    <div class="s2t-header">
      <div class="s2t-title">Sign<span>ToText</span></div>
      <div class="s2t-status">
        <span class="s2t-status-dot lost" id="s2t-status-dot"></span>
        <span id="s2t-status-text">Connecting</span>
      </div>
    </div>
    <div class="s2t-body">
      <div class="s2t-label">Live Captions</div>
      <div class="s2t-sentence-container" id="s2t-sentence-container">
        <span class="s2t-sentence-placeholder" id="s2t-placeholder">Captions will appear here...</span>
      </div>
      <div class="s2t-label">Word Buffer</div>
      <div class="s2t-buffer-container">
        <span class="s2t-buffer-text" id="s2t-buffer-text">-</span>
      </div>
    </div>
    <div class="s2t-footer">
      <button class="s2t-btn s2t-btn-secondary" id="s2t-clear-btn">Clear</button>
      <button class="s2t-btn s2t-btn-primary" id="s2t-back-btn">Backspace</button>
    </div>
  `;

  document.body.appendChild(overlayElement);

  statusDot = document.getElementById("s2t-status-dot");
  bufferText = document.getElementById("s2t-buffer-text");
  sentenceContainer = document.getElementById("s2t-sentence-container");
  sentencePlaceholder = document.getElementById("s2t-placeholder");

  // Wire clear/backspace buttons
  document.getElementById("s2t-clear-btn").addEventListener("click", () => {
    currentSentence = "";
    currentWordBuffer = "";
    updateUISentence("");
    updateUIBuffer("");
  });

  document.getElementById("s2t-back-btn").addEventListener("click", () => {
    handleBackspace();
  });
}

function updateStatusUI(label, className) {
  const statusLabel = document.getElementById("s2t-status-text");
  if (statusLabel) statusLabel.textContent = label;
  if (statusDot) {
    statusDot.className = `s2t-status-dot ${className}`;
  }
}

function updateUIBuffer(text) {
  if (bufferText) {
    bufferText.textContent = text || "-";
  }
}

function updateUISentence(text) {
  if (sentenceContainer) {
    if (text.length > 0) {
      sentenceContainer.textContent = text;
    } else {
      sentenceContainer.innerHTML = `<span class="s2t-sentence-placeholder" id="s2t-placeholder">Captions will appear here...</span>`;
    }
  }
}

function showErrorMessage(message) {
  if (sentenceContainer) {
    sentenceContainer.innerHTML = `<span style="color: #ef4444; font-size: 13px;">Error: ${message}</span>`;
  }
}
