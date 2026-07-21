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
// The "writer" (extension/writer.js) owns the assembled text now: it turns the
// token stream into readable, capitalized, punctuated sentences (and holds the
// optional on-device LLM seam). It is loaded async at init — guard on it being
// non-null before committing tokens.
let writer = null;
let wordSignAllowed = null;    // fusion referee (extension/fusion.js), loaded async with the writer
let currentWordBuffer = "";   // in-progress fingerspelled letters, before they commit as a word
let lastPredictedLabel = null;
let sameFrameCount = 0;
let lastCommitTime = 0;         // when the last word/sign landed — drives sentence-boundary punctuation
const LOCK_IN_FRAMES = 5; // lock a letter after 5 consecutive matching frames
const SENTENCE_PAUSE_MS = 3500; // a pause this long closes the current sentence (adds . or ?)

// Temporal spacing states
let lastPredictionTime = 0;
const VELOCITY_SPACE_THRESHOLD = 0.005;

// Word-sign states: a word commits after two consecutive agreeing
// inferences above threshold, then a cooldown so one sign = one word
const WORD_CONFIDENCE = 0.85;
const WORD_COOLDOWN_MS = 2500;
let wordCandidate = null;
let wordAgreeCount = 0;
let lastWordTime = 0;

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

  loadWriter();

  // The offscreen document does the actual init; ask background for it
  chrome.runtime.sendMessage({ type: "s2tEnsureOffscreen" }).catch(() => {});
}

// Load the writer module (an ES module) into this classic content script. MV3
// content scripts can't statically `import`, so we dynamically import the
// extension-URL of writer.js — which is why writer.js + assembler.js are listed
// in the manifest's web_accessible_resources. Best-effort: if it fails to load,
// captions simply won't assemble (logged), but nothing else breaks.
async function loadWriter() {
  if (writer) return;
  try {
    const mod = await import(chrome.runtime.getURL("writer.js"));
    let refiner = null;
    try {
      refiner = await mod.createGeminiNanoRefiner(); // null on Brave / non-Nano Chrome
    } catch {
      refiner = null;
    }
    writer = mod.createWriter({ refiner });
  } catch (e) {
    console.error("[SignToText] failed to load writer:", e);
  }
  try {
    const fusion = await import(chrome.runtime.getURL("fusion.js"));
    wordSignAllowed = fusion.wordSignAllowed;
  } catch (e) {
    console.error("[SignToText] failed to load fusion referee:", e);
  }
}

function stopSignToText() {
  if (overlayElement) {
    overlayElement.remove();
    overlayElement = null;
  }
  if (writer) writer.clear();
  currentWordBuffer = "";
  lastPredictedLabel = null;
  sameFrameCount = 0;
}

// ---- Prediction stream -> letters -> words ----

function handleFrame(frame) {
  // Live feedback: always show the model's best guess, even below the
  // lock-in threshold — so a near-miss looks different from a blank stare.
  const guessEl = document.getElementById("s2t-guess");
  if (guessEl) {
    if (frame.handPresent) {
      const letter = `${frame.label} ${Math.round(frame.confidence * 100)}%`;
      const word = frame.word
        ? ` · ${frame.word} ${Math.round(frame.wordConfidence * 100)}%`
        : "";
      guessEl.textContent = letter + word;
    } else {
      guessEl.textContent = "—";
    }
  }

  if (!frame.handPresent) {
    wordCandidate = null;
    wordAgreeCount = 0;
    updateStatusUI("No Hands", "lost");
    // Hand dropped: a pause boundary for word spacing
    if (currentWordBuffer.length > 0 &&
        Date.now() - lastPredictionTime >= spacingDelay) {
      commitWord();
    }
    // A longer pause closes the sentence — the writer capitalizes it and adds
    // terminal punctuation (. or ? for a question word). endSentence() is a
    // no-op once the sentence is already closed, so calling it each idle frame
    // is safe.
    if (writer && Date.now() - lastCommitTime >= SENTENCE_PAUSE_MS) {
      const finished = writer.endSentence();
      if (finished) renderSentenceUI();
    }
    return;
  }

  updateStatusUI("Tracking", "tracking");

  // Whole-word signs take precedence over fingerspelling — but the fusion
  // referee (fusion.js, T18) muzzles the word model while the user is actively
  // spelling, so a name like K-A-T-T-I isn't hijacked by a stray whole word.
  if (frame.word && frame.wordConfidence >= WORD_CONFIDENCE) {
    if (frame.word === wordCandidate) {
      wordAgreeCount++;
    } else {
      wordCandidate = frame.word;
      wordAgreeCount = 1;
    }
    if (wordAgreeCount >= 2 && Date.now() - lastWordTime >= WORD_COOLDOWN_MS) {
      // If the referee hasn't loaded yet, fall back to the old behaviour (allow).
      const allowed = !wordSignAllowed || wordSignAllowed({
        bufferLength: currentWordBuffer.length,
        msSinceLastLetter: Date.now() - lastPredictionTime,
        wordConfidence: frame.wordConfidence,
      });
      if (allowed) commitWordSign(frame.word);
    }
  }

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

// A recognized whole-word sign: replaces any half-spelled letter buffer
// (the signing motion usually litters it with stray letters). Signed words come
// from the trusted vocab, so the writer never autocorrects them.
function commitWordSign(word) {
  if (!writer) return;
  currentWordBuffer = "";
  updateUIBuffer("");
  writer.addSignedWord(word);
  lastCommitTime = Date.now();
  renderSentenceUI();
  lastWordTime = Date.now();
  wordCandidate = null;
  wordAgreeCount = 0;
  if (chatInjectionEnabled) {
    injectIntoChat(writer.text);
  }
}

function commitWord() {
  if (currentWordBuffer.length === 0) return;

  if (writer) {
    writer.addSpelledWord(currentWordBuffer); // fingerspelled → conservative autocorrect
    lastCommitTime = Date.now();
  }
  currentWordBuffer = "";

  renderSentenceUI();
  updateUIBuffer("");

  if (chatInjectionEnabled && writer) {
    injectIntoChat(writer.text);
  }
}

function appendSpace() {
  // A SPACE sign just ends the current fingerspelled word — the assembler joins
  // words with spaces itself, so there is no separate space to append.
  commitWord();
}

function handleBackspace() {
  if (currentWordBuffer.length > 0) {
    currentWordBuffer = currentWordBuffer.slice(0, -1);
    updateUIBuffer(currentWordBuffer);
  } else if (writer) {
    writer.backspace(); // drops the last assembled word
    renderSentenceUI();
  }
}

// Single place that pushes the writer's assembled text into the overlay.
function renderSentenceUI() {
  updateUISentence(writer ? writer.text : "");
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
      <div class="s2t-title">Talk Through <span>Me</span></div>
      <div class="s2t-status">
        <span class="s2t-status-dot lost" id="s2t-status-dot"></span>
        <span id="s2t-status-text">Connecting</span>
      </div>
    </div>
    <div class="s2t-body">
      <div class="s2t-sentence-container" id="s2t-sentence-container">
        <span class="s2t-sentence-placeholder" id="s2t-placeholder">Sign to see captions here&hellip;</span>
      </div>
      <details class="s2t-details">
        <summary>Details</summary>
        <div class="s2t-detail-row">
          <span class="s2t-detail-label">Word buffer</span>
          <span class="s2t-buffer-text" id="s2t-buffer-text">-</span>
        </div>
        <div class="s2t-detail-row">
          <span class="s2t-detail-label">Model sees</span>
          <span class="s2t-buffer-text" id="s2t-guess" style="opacity:0.75">&mdash;</span>
        </div>
      </details>
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
    if (writer) writer.clear();
    currentWordBuffer = "";
    renderSentenceUI();
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
