// content.js - SignToText captions overlay (UI only).
//
// All heavy machinery (webcam, MediaPipe, TF.js) runs in the extension's
// offscreen document — MediaPipe's WASM cannot load inside a content
// script (isolated world + page CSP). This script just renders the
// overlay and turns the incoming prediction stream into sentences.

let extensionEnabled = false;
let chatInjectionEnabled = false;
let confidenceThreshold = 0.90;
let spacingDelay = 750;

// UI Elements
let overlayElement = null;
let statusDot = null;
let bufferText = null;
let sentenceContainer = null;
let sentencePlaceholder = null;

// Sentence states
let currentSentence = "";
let currentWordBuffer = "";
let lastPredictedLabel = null;
let sameFrameCount = 0;
const LOCK_IN_FRAMES = 5; // lock a letter after 5 consecutive matching frames

// Temporal spacing states
let lastPredictionTime = 0;
const VELOCITY_SPACE_THRESHOLD = 0.005;

// ---- Messages from background (settings + prediction stream) ----

chrome.runtime.onMessage.addListener((request) => {
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
    return;
  }

  if (!extensionEnabled || !overlayElement) return;

  if (request.type === "s2tStatus") {
    if (request.status === "ready") {
      updateStatusUI("Ready", "lost");
    } else if (request.status === "error") {
      updateStatusUI("Init Failed", "lost");
      showErrorMessage(
        request.code === "camera-denied"
          ? "Camera permission needed — see the tab that just opened."
          : request.message || "Failed to start camera or models."
      );
    }
    return;
  }

  if (request.type === "s2tFrame") {
    handleFrame(request);
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

  console.log("[SignToText] content script loaded. enabled=", extensionEnabled);

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

function initSignToText() {
  if (overlayElement) return; // already running

  createOverlayDOM();
  updateStatusUI("Initializing...", "lost");

  // The offscreen document does the actual init; ask background for it
  chrome.runtime.sendMessage({ type: "s2tEnsureOffscreen" }).catch(() => {});
}

function stopSignToText() {
  if (overlayElement) {
    overlayElement.remove();
    overlayElement = null;
  }
  currentSentence = "";
  currentWordBuffer = "";
  lastPredictedLabel = null;
  sameFrameCount = 0;
}

// ---- Prediction stream -> letters -> words ----

function handleFrame(frame) {
  if (!frame.handPresent) {
    updateStatusUI("No Hands", "lost");
    // Hand dropped: a pause boundary for word spacing
    if (currentWordBuffer.length > 0 &&
        Date.now() - lastPredictionTime >= spacingDelay) {
      commitWord();
    }
    return;
  }

  updateStatusUI("Tracking", "tracking");

  if (frame.confidence < confidenceThreshold) return;

  if (frame.label === lastPredictedLabel) {
    sameFrameCount++;
    if (sameFrameCount === LOCK_IN_FRAMES) {
      handleLockedPrediction(frame.label, frame.wristVelocity);
    }
  } else {
    lastPredictedLabel = frame.label;
    sameFrameCount = 0;
  }
}

function handleLockedPrediction(character, wristVelocity) {
  const now = Date.now();

  // Debounce: one character per hold, not a stream
  if (now - lastPredictionTime < 1200) return;
  lastPredictionTime = now;

  if (character === "SPACE") {
    appendSpace();
  } else if (character === "DELETE") {
    handleBackspace();
  } else {
    currentWordBuffer += character;
    updateUIBuffer(currentWordBuffer);

    // Auto-spacing: stationary hand after a letter ends the word
    if (wristVelocity < VELOCITY_SPACE_THRESHOLD) {
      setTimeout(() => {
        if (Date.now() - lastPredictionTime >= spacingDelay && currentWordBuffer.length > 0) {
          commitWord();
        }
      }, spacingDelay);
    }
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

// ---- UI rendering ----

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
    sentenceContainer.innerHTML = "";
    const span = document.createElement("span");
    span.style.cssText = "color: #ef4444; font-size: 13px;";
    span.textContent = `Error: ${message}`;
    sentenceContainer.appendChild(span);
  }
}
