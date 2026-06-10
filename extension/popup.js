// popup.js - Interactive Controls for SignToText

document.addEventListener("DOMContentLoaded", () => {
  const enableToggle = document.getElementById("enable-toggle");
  const chatToggle = document.getElementById("chat-toggle");
  const thresholdSlider = document.getElementById("threshold-slider");
  const delaySlider = document.getElementById("delay-slider");
  
  const thresholdVal = document.getElementById("threshold-val");
  const delayVal = document.getElementById("delay-val");
  const statusPulse = document.getElementById("status-pulse");
  const statusText = document.getElementById("status-text");

  // Load saved configuration from chrome.storage
  chrome.storage.local.get([
    "extensionEnabled",
    "chatInjectionEnabled",
    "detectionThreshold",
    "spacingDelay"
  ], (data) => {
    // Enable state
    enableToggle.checked = data.extensionEnabled || false;
    updateStatusUI(enableToggle.checked);

    // Chat toggle
    chatToggle.checked = data.chatInjectionEnabled || false;

    // Threshold
    const thresh = Math.round((data.detectionThreshold || 0.90) * 100);
    thresholdSlider.value = thresh;
    thresholdVal.textContent = `${thresh}%`;

    // Spacing delay
    const delay = data.spacingDelay || 750;
    delaySlider.value = delay;
    delayVal.textContent = `${delay}ms`;
  });

  // Listeners for Interactive UI Elements

  enableToggle.addEventListener("change", (e) => {
    const isEnabled = e.target.checked;
    chrome.storage.local.set({ extensionEnabled: isEnabled });
    updateStatusUI(isEnabled);
  });

  chatToggle.addEventListener("change", (e) => {
    chrome.storage.local.set({ chatInjectionEnabled: e.target.checked });
  });

  thresholdSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    thresholdVal.textContent = `${val}%`;
    chrome.storage.local.set({ detectionThreshold: val / 100 });
  });

  delaySlider.addEventListener("input", (e) => {
    const val = e.target.value;
    delayVal.textContent = `${val}ms`;
    chrome.storage.local.set({ spacingDelay: parseInt(val, 10) });
  });

  // UI Status Indicator Update
  function updateStatusUI(active) {
    if (active) {
      statusPulse.className = "pulse-dot active";
      statusText.textContent = "Active";
      statusText.style.color = "hsl(145, 85%, 50%)";
    } else {
      statusPulse.className = "pulse-dot idle";
      statusText.textContent = "Idle";
      statusText.style.color = "hsl(240, 6%, 70%)";
    }
  }
});
