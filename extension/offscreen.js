// offscreen.js — camera + MediaPipe + TF.js, running in the offscreen
// document. Streams one message per analyzed frame to the rest of the
// extension; background.js relays them to content scripts.

import { FilesetResolver, HandLandmarker } from "./lib/vision_bundle.js";
import { createClassifier, argmax } from "./inference.js";

const FRAME_INTERVAL_MS = 50; // ~20 fps analysis

let landmarker = null;
let classify = null;
let labelMap = [];
let videoElement = null;
let lastWristPosition = null;
let wristVelocityRolling = 0;

function send(payload) {
  chrome.runtime.sendMessage(payload).catch(() => {});
}

function getNormalizedLandmarks(landmarks) {
  // MUST match pipeline/normalize.py exactly
  const wrist = landmarks[0];
  const mcp9 = landmarks[9];
  const scale = Math.sqrt(
    (mcp9.x - wrist.x) ** 2 + (mcp9.y - wrist.y) ** 2 + (mcp9.z - wrist.z) ** 2
  ) || 1e-6;

  const normalized = [];
  for (const lm of landmarks) {
    normalized.push(
      (lm.x - wrist.x) / scale,
      (lm.y - wrist.y) / scale,
      (lm.z - wrist.z) / scale
    );
  }
  return normalized;
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
    send({ type: "s2tFrame", handPresent: false });
    return;
  }

  const handLms = result.landmarks[0];
  trackWristVelocity(handLms[0]);
  const normLms = getNormalizedLandmarks(handLms);

  const scores = classify(normLms);
  const maxIdx = argmax(scores);

  send({
    type: "s2tFrame",
    handPresent: true,
    label: labelMap[maxIdx],
    confidence: scores[maxIdx],
    wristVelocity: wristVelocityRolling
  });
}

init();
