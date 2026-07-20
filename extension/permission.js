// permission.js — one-time camera grant for the extension origin.
// Once granted here, the offscreen document can open the camera silently.
// After a successful grant we tell the background to (re)start the engine and
// auto-close this tab, so the user never has to manually toggle off/on again.
document.getElementById("grant").addEventListener("click", async () => {
  const status = document.getElementById("status");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    stream.getTracks().forEach(t => t.stop()); // we only needed the grant
    status.textContent = "Camera granted — starting Talk Through Me…";
    // Kick the background to restart the offscreen engine now that the camera
    // is allowed, then close this tab for the user.
    chrome.runtime.sendMessage({ type: "s2tPermissionGranted" });
    setTimeout(() => window.close(), 1200);
  } catch (err) {
    status.textContent =
      "Camera access was denied. Talk Through Me can't work without it. (" + err.name + ")";
  }
});
