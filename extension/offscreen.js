// offscreen.js — camera + MediaPipe + TF.js, running in the offscreen
// document. Streams one message per analyzed frame to the rest of the
// extension; background.js relays them to content scripts.

import { FilesetResolver, HandLandmarker } from "./lib/vision_bundle.js";
import { createClassifier, createSequenceClassifier, argmax } from "./inference.js";
import { normalizeLandmarks, buildWordSequence } from "./features.js";

const FRAME_INTERVAL_MS = 50; // ~20 fps analysis

let landmarker = null;
let classify = null;
let labelMap = [];
let videoElement = null;
let lastWristPosition = null;
let wristVelocityRolling = 0;

// Word model: rolling buffer of the last WORD_FRAMES analyzed frames.
// Features per frame match pipeline/word_features.py: 63 normalized
// landmarks + 3 wrist trajectory (relative to the buffer's first frame)
// + 2 absolute wrist location (x,y in the camera frame) = 68 features.
const WORD_FRAMES = 30;
const WORD_EVERY_N_FRAMES = 8; // LSTM is heavier — run at ~2.5 Hz, not 20
let classifyWords = null;
let wordLabels = [];
let frameBuffer = [];   // { norm: number[63], wrist: {x,y,z} }
let frameCounter = 0;
let lastWordResult = null;

function send(payload) {
  chrome.runtime.sendMessage(payload).catch(() => {});
}

function trackWristVelocity(wrist) {
  if (lastWristPosition) {
    const dist = Math.sqrt(
      (wrist.x - lastWristPosition.x) ** 2 +
      (wrist.y - lastWristPosition.y) ** 2 +
      (wrist.z - lastWristPosition.z) ** 2
    );
    wristVelocityRolling = wristVelocityRolling * 0.7 + dist * 0.3;
  }
  lastWristPosition = { x: wrist.x, y: wrist.y, z: wrist.z };
}

async function createLandmarker(delegate) {
  const vision = await FilesetResolver.forVisionTasks(
    chrome.runtime.getURL("wasm")
  );
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: chrome.runtime.getURL("model/hand_landmarker.task"),
      delegate
    },
    runningMode: "VIDEO",
    numHands: 1
  });
}

async function init() {
  try {
    // 1. MediaPipe — GPU first, CPU fallback (GPU contexts can be flaky)
    try {
      landmarker = await createLandmarker("GPU");
    } catch (gpuErr) {
      console.warn("[SignToText] GPU delegate failed, falling back to CPU:", gpuErr);
      landmarker = await createLandmarker("CPU");
    }

    // 2. Classifier from exported weights — plain JS, no TF.js
    // (MV3 forbids the eval() that tf.js needs at startup)
    const resp = await fetch(chrome.runtime.getURL("model/weights.json"));
    const spec = await resp.json();
    classify = createClassifier(spec);
    labelMap = spec.labels;

    // 2b. Word LSTM (optional — extension still works letters-only if absent)
    try {
      const wResp = await fetch(chrome.runtime.getURL("model/words.json"));
      const wSpec = await wResp.json();
      classifyWords = createSequenceClassifier(wSpec);
      wordLabels = wSpec.labels;
      console.log(`[SignToText] word model loaded: ${wordLabels.length} words`);
    } catch (e) {
      console.warn("[SignToText] no word model bundled, letters only:", e);
    }

    // 3. Camera
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, frameRate: { ideal: 30 } }
    });
    videoElement = document.createElement("video");
    videoElement.srcObject = stream;
    await videoElement.play();

    send({ type: "s2tStatus", status: "ready" });

    // rAF doesn't fire in offscreen documents (never rendered) — use a timer
    setInterval(processFrame, FRAME_INTERVAL_MS);
  } catch (err) {
    console.error("[SignToText] offscreen init failed:", err);
    const code = err && err.name === "NotAllowedError" ? "camera-denied" : "init-failed";
    send({ type: "s2tStatus", status: "error", code, message: String(err && err.message || err) });
  }
}

function processFrame() {
  if (!videoElement || videoElement.readyState < 2) return;

  const result = landmarker.detectForVideo(videoElement, performance.now());

  if (!result.landmarks || result.landmarks.length === 0) {
    lastWristPosition = null;
    frameBuffer = [];        // a word sign can't span a hand-loss gap
    lastWordResult = null;
    send({ type: "s2tFrame", handPresent: false });
    return;
  }

  const handLms = result.landmarks[0];
  trackWristVelocity(handLms[0]);
  const normLms = normalizeLandmarks(handLms);

  const scores = classify(normLms);
  const maxIdx = argmax(scores);

  // Word model: feed the rolling buffer, infer every Nth frame once full
  frameBuffer.push({
    norm: normLms,
    wrist: { x: handLms[0].x, y: handLms[0].y, z: handLms[0].z }
  });
  if (frameBuffer.length > WORD_FRAMES) frameBuffer.shift();
  frameCounter++;

  if (classifyWords && frameBuffer.length === WORD_FRAMES &&
      frameCounter % WORD_EVERY_N_FRAMES === 0) {
    const seq = buildWordSequence(frameBuffer);
    const wScores = classifyWords(seq);
    const wIdx = argmax(wScores);
    lastWordResult = { label: wordLabels[wIdx], confidence: wScores[wIdx] };
  }

  send({
    type: "s2tFrame",
    handPresent: true,
    label: labelMap[maxIdx],
    confidence: scores[maxIdx],
    wristVelocity: wristVelocityRolling,
    word: lastWordResult ? lastWordResult.label : null,
    wordConfidence: lastWordResult ? lastWordResult.confidence : 0
  });
}

init();
