// permission.js — one-time camera grant for the extension origin.
// Once granted here, the offscreen document can open the camera silently.
document.getElementById("grant").addEventListener("click", async () => {
  const status = document.getElementById("status");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    stream.getTracks().forEach(t => t.stop()); // we only needed the grant
    status.textContent = "Camera access granted! Toggle SignToText off and on again, then close this tab.";
  } catch (err) {
    status.textContent = "Camera access was denied. SignToText cannot work without it. (" + err.name + ")";
  }
});
