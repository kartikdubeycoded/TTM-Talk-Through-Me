// features.js — the feature math shared by the live offscreen document and the
// Node parity test (tests/smoke_features.mjs). Pure functions, no browser APIs,
// so the test can import them and verify they match the Python training
// pipeline (pipeline/normalize.py + pipeline/word_features.py). Keeping ONE
// implementation behind a test is the whole point: the live extension must feed
// the model the same numbers it trained on.

// 21 hand landmarks -> 63 normalized numbers. MUST match the wrist-anchor +
// knuckle-scale convention in pipeline/normalize.py.
export function normalizeLandmarks(landmarks) {
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

// Rolling buffer of { norm: number[63], wrist: {x,y,z} } -> (T, 68) features.
// MUST match pipeline/word_features.py sequence_features():
//   [0:63]  normalized landmarks
//   [63:66] wrist trajectory: this frame's wrist minus frame 0's
//   [66:68] wrist location: absolute x,y in the camera frame
export function buildWordSequence(buffer) {
  const w0 = buffer[0].wrist;
  return buffer.map((f) => [
    ...f.norm,
    f.wrist.x - w0.x, f.wrist.y - w0.y, f.wrist.z - w0.z,  // [63:66] trajectory
    f.wrist.x, f.wrist.y,                                   // [66:68] location
  ]);
}
